//
// BlackjackBridgeApp.swift
//
// Entry point of the bridge app: Meta glasses -> iPhone -> PC. The iPhone does no
// vision work. It forwards the glasses' video to the PC and speaks the PC's commands.
//

import MWDATCore
import SwiftUI

@main
struct BlackjackBridgeApp: App {
    @State private var model = BridgeModel()

    var body: some Scene {
        WindowGroup {
            ContentView(model: model)
                // Meta AI calls the app back through its URL scheme after registration and
                // permission requests. Only DAT links are forwarded to the SDK.
                .onOpenURL { url in
                    guard
                        let components = URLComponents(url: url, resolvingAgainstBaseURL: false),
                        components.queryItems?.contains(where: { $0.name == "metaWearablesAction" }) == true
                    else {
                        model.log.add("Ignored URL: \(url.absoluteString)")
                        return
                    }
                    Task {
                        do {
                            _ = try await Wearables.shared.handleUrl(url)
                            model.log.add("Meta AI callback handled")
                        } catch {
                            model.log.add("Meta AI callback failed: \(error.localizedDescription)")
                        }
                    }
                }
        }
    }
}
