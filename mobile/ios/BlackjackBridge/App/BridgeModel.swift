//
// BridgeModel.swift
//
// App state shared by the UI: PC address, stream settings, connection status and the
// on-screen log. Glasses streaming (step 2), the TCP client (step 3) and speech (step 4)
// plug into this model.
//

import Foundation
import MWDATCore
import Observation

@MainActor
@Observable
final class BridgeModel {
    /// Connection state shown on screen.
    enum ConnectionState: String {
        case disconnected
        case connecting
        case connected
    }

    private enum DefaultsKey {
        static let serverAddress = "serverAddress"
        static let resolution = "streamResolution"
        static let frameRate = "streamFrameRate"
    }

    let log = LogStore()

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

    var connectionState: ConnectionState = .disconnected
    var framesSent = 0
    var sendFps = 0.0
    var lastCommand: String?

    var serverAddress: ServerAddress? { ServerAddress(serverAddressText) }

    let appVersion: String = {
        let info = Bundle.main.infoDictionary
        let version = info?["CFBundleShortVersionString"] as? String ?? "?"
        let build = info?["CFBundleVersion"] as? String ?? "?"
        return "\(version) (\(build))"
    }()

    init() {
        let defaults = UserDefaults.standard
        serverAddressText = defaults.string(forKey: DefaultsKey.serverAddress) ?? ""
        resolution = defaults.string(forKey: DefaultsKey.resolution)
            .flatMap(StreamResolution.init(rawValue:)) ?? StreamSettings.defaultResolution
        let storedRate = UInt(defaults.integer(forKey: DefaultsKey.frameRate))
        frameRate = StreamSettings.frameRates.contains(storedRate) ? storedRate : StreamSettings.defaultFrameRate

        log.add("BJ Bridge \(appVersion) started")
        do {
            try Wearables.configure()
            log.add("Wearables SDK configured")
        } catch {
            log.add("Wearables SDK configuration failed: \(error.localizedDescription)")
        }
    }

    /// Placeholder until the TCP client exists (step 3).
    func toggleConnection() {
        guard let address = serverAddress else {
            log.add("Invalid PC address \"\(serverAddressText)\". Use host:port, e.g. 192.168.1.50:5050")
            return
        }
        log.add("Connect to \(address) requested. The TCP client is not implemented yet")
    }
}
