//
// GlassesController.swift
//
// Talks to the Meta glasses through the Wearables DAT SDK: registration with Meta AI,
// the device session and the camera stream. It follows the lifecycle of Meta's
// CameraAccess sample: register -> start session -> check camera permission -> add the
// camera -> start the stream. Frames arrive as raw (decoded) images and are exposed for
// the on-screen preview. From step 3 they are also handed to the TCP client.
//

import Foundation
import MWDATCamera
import MWDATCore
import Observation
import UIKit

@MainActor
@Observable
final class GlassesController {
    // MARK: - Observable state

    private(set) var registrationState: RegistrationState
    private(set) var hasActiveDevice = false
    private(set) var sessionState: DeviceSessionState = .idle
    private(set) var streamState: StreamState = .stopped

    /// Latest frame for the on-screen preview.
    private(set) var currentFrame: UIImage?
    /// Size of the frames actually delivered (the SDK can lower it on its own).
    private(set) var frameSize: CGSize?
    private(set) var framesReceived = 0
    /// Frames received per second, averaged over the last two seconds.
    private(set) var receivedFps = 0.0

    var isRegistered: Bool { registrationState == .registered }
    var hasSession: Bool {
        switch sessionState {
        case .starting, .started, .paused, .stopping: true
        default: false
        }
    }
    var isSessionStarted: Bool { sessionState == .started }
    /// A stream is attached (in any state other than stopped).
    var hasStream: Bool { streamState != .stopped }

    // MARK: - Private

    @ObservationIgnored private let wearables: WearablesInterface
    @ObservationIgnored private let deviceSelector: AutoDeviceSelector
    @ObservationIgnored private let log: LogStore
    @ObservationIgnored private var deviceSession: DeviceSession?
    @ObservationIgnored private var camera: MWDATCamera.Camera?
    @ObservationIgnored private let sessionTokens = ListenerTokenBag()
    @ObservationIgnored private let streamTokens = ListenerTokenBag()
    @ObservationIgnored private var registrationTask: Task<Void, Never>?
    @ObservationIgnored private var deviceTask: Task<Void, Never>?
    @ObservationIgnored private var frameTimes: [TimeInterval] = []

    init(wearables: WearablesInterface, log: LogStore) {
        self.wearables = wearables
        self.log = log
        self.deviceSelector = AutoDeviceSelector(wearables: wearables)
        self.registrationState = wearables.registrationState
        log.add("Registration: \(String(describing: wearables.registrationState))")

        registrationTask = Task { [weak self] in
            for await state in wearables.registrationStateStream() {
                guard let self else { return }
                if state != self.registrationState {
                    self.log.add("Registration: \(String(describing: state))")
                }
                self.registrationState = state
            }
        }

        let selector = deviceSelector
        deviceTask = Task { [weak self] in
            for await deviceId in selector.activeDeviceStream() {
                guard let self else { return }
                let active = deviceId != nil
                if active != self.hasActiveDevice {
                    self.log.add(active ? "Glasses available" : "No active glasses")
                }
                self.hasActiveDevice = active
            }
        }
    }

    // MARK: - Registration

    /// Opens Meta AI so the user approves the app. Meta AI calls back through the URL scheme.
    func connectGlasses() {
        guard registrationState != .registering else { return }
        log.add("Opening Meta AI to register the app")
        Task {
            do {
                try await wearables.startRegistration()
            } catch {
                log.add("Registration failed: \(error.localizedDescription)")
            }
        }
    }

    func disconnectGlasses() {
        stopStreaming()
        endSession()
        Task {
            do {
                try await wearables.startUnregistration()
            } catch {
                log.add("Unregistration failed: \(error.localizedDescription)")
            }
        }
    }

    // MARK: - Session

    func startSession() {
        guard !hasSession else { return }
        do {
            let session = try wearables.createSession(deviceSelector: deviceSelector)
            deviceSession = session
            observeSession(session)
            sessionState = .starting
            log.add("Starting session")
            try session.start()
        } catch {
            log.add("Session failed to start: \(error.localizedDescription)")
            sessionState = .idle
            cleanupSession()
        }
    }

    func endSession() {
        guard let session = deviceSession else { return }
        log.add("Ending session")
        sessionState = .stopping
        session.stop()
    }

    private func observeSession(_ session: DeviceSession) {
        session.statePublisher.listen { [weak self] state in
            Task { @MainActor in self?.handleSessionState(state) }
        }.store(in: sessionTokens)
        session.errorPublisher.listen { [weak self] error in
            let message = error.localizedDescription
            Task { @MainActor in self?.log.add("Session error: \(message)") }
        }.store(in: sessionTokens)
    }

    private func handleSessionState(_ state: DeviceSessionState) {
        sessionState = state
        log.add("Session: \(String(describing: state))")
        if state == .stopped {
            clearStream()
            cleanupSession()
        }
    }

    private func cleanupSession() {
        sessionTokens.clear()
        deviceSession = nil
    }

    // MARK: - Stream

    /// Starts the camera stream, asking Meta AI for camera permission first if needed.
    func startStreaming(configuration: StreamConfiguration) async {
        guard let session = deviceSession, session.state == .started else {
            log.add("Start a session before starting the stream")
            return
        }
        guard camera == nil else { return }
        do {
            if try await wearables.checkPermissionStatus(.camera) != .granted {
                log.add("Opening Meta AI to grant camera permission")
                guard try await wearables.requestPermission(.camera) == .granted else {
                    log.add("Camera permission denied")
                    return
                }
            }
        } catch {
            log.add("Camera permission check failed: \(error.localizedDescription)")
            return
        }
        beginStream(on: session, configuration: configuration)
    }

    func stopStreaming() {
        guard let activeCamera = camera else { return }
        log.add("Stopping stream")
        streamState = .stopping
        activeCamera.stop()
    }

    private func beginStream(on session: DeviceSession, configuration: StreamConfiguration) {
        guard camera == nil else { return }
        do {
            guard let newCamera = try session.addCamera(config: configuration) else {
                log.add("Couldn't create the camera, try again")
                return
            }
            camera = newCamera
            // Subscribe before start() so no state transition is missed.
            observeStream(newCamera.stream)
            streamState = .starting
            log.add("Starting stream")
            newCamera.stream.start()
        } catch {
            camera = nil
            log.add("Couldn't add the camera: \(error.localizedDescription)")
        }
    }

    private func observeStream(_ stream: MWDATCamera.Stream) {
        stream.statePublisher.listen { [weak self] state in
            Task { @MainActor in self?.handleStreamState(state) }
        }.store(in: streamTokens)

        stream.videoFramePublisher.listen { [weak self] frame in
            // Called off the main thread. Decode here and hop to the main actor with the
            // image only. UIImage is immutable, so handing it over is safe.
            nonisolated(unsafe) let image = frame.makeUIImage()
            let receivedAt = Date().timeIntervalSince1970
            Task { @MainActor in self?.handleFrame(image, receivedAt: receivedAt) }
        }.store(in: streamTokens)

        stream.errorPublisher.listen { [weak self] error in
            let message = error.localizedDescription
            Task { @MainActor in self?.log.add("Stream error: \(message)") }
        }.store(in: streamTokens)
    }

    private func handleStreamState(_ state: StreamState) {
        streamState = state
        log.add("Stream: \(String(describing: state))")
        // Keep the screen on while the glasses are streaming.
        UIApplication.shared.isIdleTimerDisabled = state == .streaming
        if state == .stopped {
            clearStream()
        }
    }

    private func handleFrame(_ image: UIImage?, receivedAt: TimeInterval) {
        guard let image, camera != nil else { return }
        currentFrame = image
        framesReceived += 1

        let size = CGSize(width: image.size.width * image.scale, height: image.size.height * image.scale)
        if size != frameSize {
            log.add("Frame size: \(Int(size.width))x\(Int(size.height))")
            frameSize = size
        }

        frameTimes.append(receivedAt)
        frameTimes.removeAll { receivedAt - $0 > 2 }
        if let first = frameTimes.first, frameTimes.count > 1, receivedAt > first {
            receivedFps = Double(frameTimes.count - 1) / (receivedAt - first)
        }
    }

    /// Single teardown point for the stream (user stop, SDK stop or session end).
    private func clearStream() {
        streamTokens.clear()
        camera?.stop()
        camera = nil
        streamState = .stopped
        currentFrame = nil
        frameSize = nil
        frameTimes.removeAll()
        receivedFps = 0
        UIApplication.shared.isIdleTimerDisabled = false
    }
}
