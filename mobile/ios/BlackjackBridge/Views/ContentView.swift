//
// ContentView.swift
//
// Single screen: PC address, stream settings, status and the on-screen log panel.
//

import MWDATCore
import SwiftUI
import UIKit

struct ContentView: View {
    @Bindable var model: BridgeModel

    var body: some View {
        NavigationStack {
            Form {
                glassesSection
                pcSection
                streamSection
                statusSection
                logSection
            }
            .navigationTitle("BJ Bridge")
        }
    }

    private var glassesSection: some View {
        let glasses = model.glasses
        return Section("Glasses") {
            GlassesPreview(glasses: glasses)
                .listRowInsets(EdgeInsets())

            if !glasses.isRegistered {
                Button("Connect glasses (Meta AI)") { glasses.connectGlasses() }
                    .disabled(glasses.registrationState == .registering)
            } else if !glasses.hasSession {
                Button("Start session") { glasses.startSession() }
                    .disabled(!glasses.hasActiveDevice)
            } else {
                if glasses.hasStream {
                    Button("Stop stream") { glasses.stopStreaming() }
                } else {
                    Button("Start stream") { model.startGlassesStream() }
                        .disabled(!glasses.isSessionStarted)
                }
                Button("End session", role: .destructive) { glasses.endSession() }
            }

            LabeledContent("Registration", value: String(describing: glasses.registrationState))
            LabeledContent("Glasses", value: glasses.hasActiveDevice ? "available" : "not found")
            LabeledContent("Session", value: String(describing: glasses.sessionState))
            LabeledContent("Stream", value: String(describing: glasses.streamState))
            LabeledContent("Frame size", value: glasses.frameSize.map { "\(Int($0.width))×\(Int($0.height))" } ?? "–")
            LabeledContent(
                "Received",
                value: String(format: "%.1f fps", glasses.receivedFps) + " (\(glasses.framesReceived) frames)"
            )
        }
    }

    private var pcSection: some View {
        let isDisconnected = model.connectionState == .disconnected
        return Section("PC") {
            TextField("192.168.1.50:5050", text: $model.serverAddressText)
                .keyboardType(.numbersAndPunctuation)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
                .font(.body.monospaced())
                // Editing the address only applies on the next connect.
                .disabled(!isDisconnected)
            Button(isDisconnected ? "Connect" : "Disconnect") {
                model.toggleConnection()
            }
            .disabled(isDisconnected && model.serverAddress == nil)
        }
    }

    private var streamSection: some View {
        Section("Glasses stream") {
            Picker("Resolution", selection: $model.resolution) {
                ForEach(StreamResolution.allCases) { resolution in
                    Text(resolution.label).tag(resolution)
                }
            }
            Picker("Frame rate", selection: $model.frameRate) {
                ForEach(StreamSettings.frameRates, id: \.self) { rate in
                    Text("\(rate) fps").tag(rate)
                }
            }
        }
        // The configuration is fixed when the stream starts.
        .disabled(model.glasses.hasStream)
    }

    private var statusSection: some View {
        Section("Status") {
            LabeledContent("PC", value: model.connectionState.rawValue)
            LabeledContent("Frames sent", value: "\(model.framesSent)")
            LabeledContent("Send rate", value: String(format: "%.1f fps", model.sendFps))
            LabeledContent("Last command", value: model.lastCommand ?? "–")
            LabeledContent("App version", value: model.appVersion)
        }
    }

    private var logSection: some View {
        Section {
            LogPanel(log: model.log)
                .frame(height: 260)
                .listRowInsets(EdgeInsets())
        } header: {
            HStack {
                Text("Log")
                Spacer()
                Button("Copy") { UIPasteboard.general.string = model.log.text }
                Button("Clear") { model.log.clear() }
            }
            .textCase(nil)
        }
    }
}

/// Live frame from the glasses, shown as delivered (portrait, not cropped).
private struct GlassesPreview: View {
    let glasses: GlassesController

    var body: some View {
        ZStack {
            Color.black
            if let frame = glasses.currentFrame {
                Image(uiImage: frame)
                    .resizable()
                    .scaledToFit()
            } else {
                Text(glasses.hasStream ? "Waiting for frames…" : "No video")
                    .font(.footnote)
                    .foregroundStyle(.white.opacity(0.7))
            }
        }
        .frame(height: 360)
    }
}

/// Scrollable monospaced log that follows the newest entry.
private struct LogPanel: View {
    let log: LogStore

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 2) {
                    ForEach(log.entries) { entry in
                        Text(entry.line)
                            .font(.caption2.monospaced())
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .id(entry.id)
                    }
                }
                .padding(8)
            }
            .background(Color.black.opacity(0.04))
            .onChange(of: log.entries.last?.id) { _, newID in
                if let newID {
                    proxy.scrollTo(newID, anchor: .bottom)
                }
            }
        }
    }
}
