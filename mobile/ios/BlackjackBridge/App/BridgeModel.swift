//
// BridgeModel.swift
//
// App state shared by the UI: PC address, stream settings, connection status and the
// on-screen log. Glasses streaming (step 2), the TCP client (step 3) and speech (step 4)
// plug into this model.
//

import Foundation
import MWDATCamera
import MWDATCore
import Observation

@MainActor
@Observable
final class BridgeModel {
    private enum DefaultsKey {
        static let serverAddress = "serverAddress"
        static let resolution = "streamResolution"
        static let frameRate = "streamFrameRate"
    }

    let log: LogStore
    let glasses: GlassesController
    let speaker: Speaker
    let pc: PCClient

    /// PC receiver as "host:port", persisted in UserDefaults.
    var serverAddressText: String {
        didSet { UserDefaults.standard.set(serverAddressText, forKey: DefaultsKey.serverAddress) }
    }

    var resolution: StreamResolution {
        didSet { UserDefaults.standard.set(resolution.rawValue, forKey: DefaultsKey.resolution) }
    }

    var frameRate: UInt {
        didSet { UserDefaults.standard.set(Int(frameRate), forKey: DefaultsKey.frameRate) }
    }

    var connectionState: PCConnectionState = .disconnected
    var framesSent = 0
    /// Frames written to the PC per second, averaged over the last two seconds.
    var sendFps = 0.0
    var lastCommand: String?
    /// Last "status" text sent by the PC (e.g. the current count or the round state).
    var statusText: String?

    var serverAddress: ServerAddress? { ServerAddress(serverAddressText) }

    /// "0.1.0 (42)": marketing version and CI run number.
    static let appVersion: String = {
        let info = Bundle.main.infoDictionary
        let version = info?["CFBundleShortVersionString"] as? String ?? "?"
        let build = info?["CFBundleVersion"] as? String ?? "?"
        return "\(version) (\(build))"
    }()

    var appVersion: String { Self.appVersion }

    @ObservationIgnored private var pcTask: Task<Void, Never>?
    @ObservationIgnored private var sentTimes: [TimeInterval] = []

    init() {
        let defaults = UserDefaults.standard
        serverAddressText = defaults.string(forKey: DefaultsKey.serverAddress) ?? ""
        let resolution = defaults.string(forKey: DefaultsKey.resolution)
            .flatMap(StreamResolution.init(rawValue:)) ?? StreamSettings.defaultResolution
        let storedRate = UInt(defaults.integer(forKey: DefaultsKey.frameRate))
        let frameRate = StreamSettings.frameRates.contains(storedRate) ? storedRate : StreamSettings.defaultFrameRate
        self.resolution = resolution
        self.frameRate = frameRate

        let log = LogStore()
        self.log = log
        log.add("BJ Bridge \(Self.appVersion) started")
        // The SDK must be configured before Wearables.shared is used.
        do {
            try Wearables.configure()
            log.add("Wearables SDK configured")
        } catch {
            log.add("Wearables SDK configuration failed: \(error.localizedDescription)")
        }

        let pc = PCClient(appVersion: Self.appVersion, stream: Self.streamInfo(resolution, frameRate))
        self.pc = pc
        glasses = GlassesController(wearables: Wearables.shared, log: log, frameSink: pc)
        speaker = Speaker(log: log)
        // Every log line also goes to the PC (dropped while disconnected).
        log.onEntry = { [weak pc] entry in pc?.sendLog(entry.line) }

        pcTask = Task { [weak self] in
            for await event in pc.events {
                guard let self else { return }
                self.handle(event)
            }
        }
    }

    /// Starts the glasses stream with the resolution and frame rate chosen on screen.
    func startGlassesStream() {
        log.add("Requesting \(resolution.label) @ \(frameRate) fps")
        pc.updateStream(Self.streamInfo(resolution, frameRate))
        let configuration = StreamSettings.configuration(resolution: resolution, frameRate: frameRate)
        Task { await glasses.startStreaming(configuration: configuration) }
    }

    /// Connects to the PC (and keeps retrying every 2 s) or disconnects.
    func toggleConnection() {
        if connectionState != .disconnected {
            pc.disconnect()
            return
        }
        guard let address = serverAddress else {
            log.add("Invalid PC address \"\(serverAddressText)\". Use host:port, e.g. 192.168.1.50:5050")
            return
        }
        pc.connect(to: address)
    }

    // MARK: - PC events

    private func handle(_ event: PCClient.Event) {
        switch event {
        case .state(let state):
            connectionState = state
            if state != .connected {
                sentTimes.removeAll()
                sendFps = 0
            }
        case .log(let message):
            log.add(message)
        case .frameSent:
            countSentFrame()
        case .command(let payload):
            handleCommand(payload)
        }
    }

    private func countSentFrame() {
        framesSent += 1
        let now = Date().timeIntervalSince1970
        sentTimes.append(now)
        sentTimes.removeAll { now - $0 > 2 }
        if let first = sentTimes.first, sentTimes.count > 1, now > first {
            sendFps = Double(sentTimes.count - 1) / (now - first)
        }
    }

    /// "say" is spoken through the glasses, "status" is shown on screen.
    private func handleCommand(_ payload: Data) {
        guard let command = try? JSONDecoder().decode(BridgeProtocol.Command.self, from: payload) else {
            log.add("Unreadable command: \(String(decoding: payload, as: UTF8.self))")
            return
        }
        let text = command.text ?? ""
        lastCommand = "\(command.type): \(text)"
        switch command.type {
        case "say":
            speaker.say(text, language: command.lang)
        case "status":
            statusText = text
            log.add("Status: \(text)")
        default:
            log.add("Unknown command type \"\(command.type)\"")
        }
    }

    private static func streamInfo(_ resolution: StreamResolution, _ frameRate: UInt) -> StreamInfo {
        StreamInfo(width: resolution.width, height: resolution.height, fps: Int(frameRate))
    }
}
