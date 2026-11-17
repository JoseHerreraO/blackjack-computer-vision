//
// ContentView.swift
//
// Single screen: PC address, stream settings, status and the on-screen log panel.
//

import SwiftUI
import UIKit

struct ContentView: View {
    @Bindable var model: BridgeModel

    var body: some View {
        NavigationStack {
            Form {
                pcSection
                streamSection
                statusSection
                logSection
            }
            .navigationTitle("BJ Bridge")
        }
    }

    private var pcSection: some View {
        Section("PC") {
            TextField("192.168.1.50:5050", text: $model.serverAddressText)
                .keyboardType(.numbersAndPunctuation)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
                .font(.body.monospaced())
            Button(model.connectionState == .disconnected ? "Connect" : "Disconnect") {
                model.toggleConnection()
            }
            .disabled(model.serverAddress == nil)
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
