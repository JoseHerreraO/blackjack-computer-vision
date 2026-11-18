//
// PCClient.swift
//
// TCP client to the PC (see BridgeProtocol.swift for the wire format). It sends HELLO on
// connect, then FRAME and LOG messages, and reads COMMAND messages back.
//
// Frames are latest-only. At most one frame is being written to the socket at a time,
// and while it is in flight newer frames replace each other in a single slot, so old
// frames are dropped instead of queued and the PC never falls behind the glasses.
//
// If the connection fails or drops, it retries every 2 s until the user disconnects.
//
// All mutable state lives on one serial queue (the same one NWConnection reports on).
// Events for the UI are delivered through `events`.
//

import Foundation
import Network
import UIKit

enum PCConnectionState: String, Sendable {
    case disconnected
    case connecting
    case connected
}

/// What HELLO announces about the stream before frames arrive.
struct StreamInfo: Equatable, Sendable {
    var width: Int
    var height: Int
    var fps: Int
}

final class PCClient: @unchecked Sendable {
    enum Event: Sendable {
        case state(PCConnectionState)
        case log(String)
        case frameSent
        case command(Data)
    }

    static let retryInterval: TimeInterval = 2

    let events: AsyncStream<Event>
    private let continuation: AsyncStream<Event>.Continuation
    private let queue = DispatchQueue(label: "es.icai.blackjackvision.bridge.pc", qos: .userInitiated)
    private let appVersion: String

    // Only touched on `queue`.
    private var address: ServerAddress?
    private var connection: NWConnection?
    /// Bumped whenever a connection is replaced, so callbacks from an old one are ignored.
    private var connectionID = 0
    private var state: PCConnectionState = .disconnected
    private var failedAttempts = 0
    private var retryWork: DispatchWorkItem?
    private var requestedStream: StreamInfo
    private var deliveredSize: (width: Int, height: Int)?
    private var announcedHello: BridgeProtocol.Hello?
    private var frameInFlight = false
    private var pendingFrame: PendingFrame?
    private var jpegFailureLogged = false

    private struct PendingFrame {
        let image: UIImage
        let capturedAtMs: Int64
    }

    init(appVersion: String, stream: StreamInfo) {
        self.appVersion = appVersion
        self.requestedStream = stream
        let (stream, continuation) = AsyncStream.makeStream(of: Event.self)
        self.events = stream
        self.continuation = continuation
    }

    deinit {
        continuation.finish()
    }

    // MARK: - Public API (any thread)

    func connect(to address: ServerAddress) {
        queue.async { [self] in
            self.address = address
            failedAttempts = 0
            cancelRetry()
            log("Connecting to \(address)")
            openConnection()
        }
    }

    func disconnect() {
        queue.async { [self] in
            guard address != nil else { return }
            address = nil
            cancelRetry()
            closeConnection()
            setState(.disconnected)
            log("Disconnected from the PC")
        }
    }

    /// Stream settings that HELLO announces until the first frame shows the real size.
    func updateStream(_ stream: StreamInfo) {
        queue.async { [self] in
            requestedStream = stream
            deliveredSize = nil
            sendHelloIfChanged()
        }
    }

    /// Queues a frame, replacing any frame that has not started sending yet.
    func submitFrame(_ image: UIImage, capturedAtMs: Int64) {
        nonisolated(unsafe) let image = image
        queue.async { [self] in
            guard state == .connected else { return }
            pendingFrame = PendingFrame(image: image, capturedAtMs: capturedAtMs)
            sendPendingFrame()
        }
    }

    func sendLog(_ line: String) {
        queue.async { [self] in
            guard state == .connected else { return }
            send(BridgeProtocol.message(.log, payload: Data(line.utf8)))
        }
    }

    // MARK: - Connection

    private func openConnection() {
        guard let address else { return }
        closeConnection()
        guard let port = NWEndpoint.Port(rawValue: address.port) else {
            log("Invalid port \(address.port)")
            return
        }

        let tcp = NWProtocolTCP.Options()
        tcp.noDelay = true
        tcp.connectionTimeout = 5
        tcp.enableKeepalive = true
        tcp.keepaliveIdle = 5
        let newConnection = NWConnection(
            host: NWEndpoint.Host(address.host),
            port: port,
            using: NWParameters(tls: nil, tcp: tcp)
        )
        connection = newConnection
        let id = connectionID
        newConnection.stateUpdateHandler = { [weak self] newState in
            self?.handle(newState, id: id)
        }
        setState(.connecting)
        newConnection.start(queue: queue)
    }

    private func handle(_ newState: NWConnection.State, id: Int) {
        guard id == connectionID else { return }
        switch newState {
        case .ready:
            failedAttempts = 0
            setState(.connected)
            announcedHello = nil
            log("Connected to the PC")
            // HELLO is the first message on every connection. Log lines go through the
            // main actor and come back later, so they always follow it.
            sendHelloIfChanged()
            receiveMessage(id: id)
        case .waiting(let error):
            // Refused or unreachable. NWConnection would only retry on a network change,
            // so retry on our own schedule instead.
            connectionLost("PC not reachable (\(error))")
        case .failed(let error):
            connectionLost("Connection failed (\(error))")
        default:
            break
        }
    }

    private func connectionLost(_ reason: String) {
        let wasConnected = state == .connected
        closeConnection()
        guard address != nil else {
            setState(.disconnected)
            return
        }
        // Log the first failure of a streak only, not every 2 s.
        if wasConnected || failedAttempts == 0 {
            log("\(reason). Retrying every \(Int(Self.retryInterval)) s")
        }
        failedAttempts += 1
        setState(.connecting)
        scheduleRetry()
    }

    private func closeConnection() {
        connectionID += 1
        connection?.stateUpdateHandler = nil
        connection?.cancel()
        connection = nil
        frameInFlight = false
        pendingFrame = nil
    }

    private func scheduleRetry() {
        cancelRetry()
        let work = DispatchWorkItem { [weak self] in self?.openConnection() }
        retryWork = work
        queue.asyncAfter(deadline: .now() + Self.retryInterval, execute: work)
    }

    private func cancelRetry() {
        retryWork?.cancel()
        retryWork = nil
    }

    // MARK: - Sending

    private func send(_ data: Data, onSent: (@Sendable () -> Void)? = nil) {
        guard let connection else { return }
        let id = connectionID
        connection.send(content: data, completion: .contentProcessed { [weak self] error in
            guard let self, id == self.connectionID else { return }
            if let error {
                self.connectionLost("Send failed (\(error))")
                return
            }
            onSent?()
        })
    }

    private func currentHello() -> BridgeProtocol.Hello {
        BridgeProtocol.Hello(
            device: BridgeProtocol.deviceDescription,
            width: deliveredSize?.width ?? requestedStream.width,
            height: deliveredSize?.height ?? requestedStream.height,
            fps: requestedStream.fps,
            jpegQuality: BridgeProtocol.jpegQuality,
            appVersion: appVersion
        )
    }

    private func sendHelloIfChanged() {
        guard state == .connected else { return }
        let hello = currentHello()
        guard hello != announcedHello else { return }
        guard let json = try? JSONEncoder().encode(hello) else { return }
        announcedHello = hello
        send(BridgeProtocol.message(.hello, payload: json))
        log("HELLO \(hello.width)x\(hello.height) @ \(hello.fps) fps")
    }

    private func sendPendingFrame() {
        guard state == .connected, !frameInFlight, let frame = pendingFrame else { return }
        pendingFrame = nil

        guard let jpeg = frame.image.jpegData(compressionQuality: BridgeProtocol.jpegQuality) else {
            if !jpegFailureLogged {
                jpegFailureLogged = true
                log("Couldn't encode a frame as JPEG")
            }
            return
        }

        // The SDK may deliver a different size than requested. Tell the PC first.
        let size = pixelSize(of: frame.image)
        if deliveredSize?.width != size.width || deliveredSize?.height != size.height {
            deliveredSize = size
            sendHelloIfChanged()
        }

        frameInFlight = true
        send(BridgeProtocol.frame(capturedAtMs: frame.capturedAtMs, jpeg: jpeg)) { [weak self] in
            guard let self else { return }
            self.frameInFlight = false
            self.continuation.yield(.frameSent)
            self.sendPendingFrame()
        }
    }

    private func pixelSize(of image: UIImage) -> (width: Int, height: Int) {
        if let cgImage = image.cgImage {
            return (cgImage.width, cgImage.height)
        }
        return (Int(image.size.width * image.scale), Int(image.size.height * image.scale))
    }

    // MARK: - Receiving

    private func receiveMessage(id: Int) {
        guard let connection, id == connectionID else { return }
        let headerLength = BridgeProtocol.headerLength
        connection.receive(minimumIncompleteLength: headerLength, maximumLength: headerLength) {
            [weak self] data, _, isComplete, error in
            guard let self, id == self.connectionID else { return }
            if let error {
                self.connectionLost("Receive failed (\(error))")
                return
            }
            guard let data, let header = BridgeProtocol.parseHeader(data) else {
                self.connectionLost(isComplete ? "The PC closed the connection" : "Bad message header")
                return
            }
            guard header.length <= BridgeProtocol.maxIncomingPayload else {
                self.connectionLost("Message from the PC too large (\(header.length) bytes)")
                return
            }
            if header.length == 0 {
                self.handleIncoming(type: header.type, payload: Data())
                self.receiveMessage(id: id)
            } else {
                self.receivePayload(type: header.type, length: header.length, id: id)
            }
        }
    }

    private func receivePayload(type: UInt8, length: Int, id: Int) {
        guard let connection, id == connectionID else { return }
        connection.receive(minimumIncompleteLength: length, maximumLength: length) {
            [weak self] data, _, _, error in
            guard let self, id == self.connectionID else { return }
            if let error {
                self.connectionLost("Receive failed (\(error))")
                return
            }
            guard let data, data.count == length else {
                self.connectionLost("The PC closed the connection")
                return
            }
            self.handleIncoming(type: type, payload: data)
            self.receiveMessage(id: id)
        }
    }

    private func handleIncoming(type: UInt8, payload: Data) {
        if type == MessageType.command.rawValue {
            continuation.yield(.command(payload))
        } else {
            log(String(format: "Ignoring message type 0x%02X from the PC", type))
        }
    }

    // MARK: - Events

    private func setState(_ newState: PCConnectionState) {
        guard newState != state else { return }
        state = newState
        continuation.yield(.state(newState))
    }

    private func log(_ message: String) {
        continuation.yield(.log(message))
    }
}
