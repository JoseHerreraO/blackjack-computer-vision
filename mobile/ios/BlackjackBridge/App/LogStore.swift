//
// LogStore.swift
//
// In-memory log shown on screen. There is no Xcode console (we have no Mac), so this
// panel is the only way to see what the app is doing. From step 3 every entry is also
// sent to the PC as a LOG message.
//

import Foundation
import Observation

@MainActor
@Observable
final class LogStore {
    struct Entry: Identifiable, Sendable {
        let id = UUID()
        let date: Date
        let message: String
        /// "HH:mm:ss.SSS message", formatted once when the entry is created.
        let line: String
    }

    private(set) var entries: [Entry] = []
    private let maxEntries = 500
    @ObservationIgnored private let timeFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm:ss.SSS"
        return formatter
    }()

    /// Called for every new entry, e.g. to forward it to the PC.
    @ObservationIgnored var onEntry: ((Entry) -> Void)?

    func add(_ message: String) {
        let date = Date()
        let entry = Entry(date: date, message: message, line: "\(timeFormatter.string(from: date)) \(message)")
        entries.append(entry)
        if entries.count > maxEntries {
            entries.removeFirst(entries.count - maxEntries)
        }
        print(entry.line)
        onEntry?(entry)
    }

    func clear() {
        entries.removeAll()
    }

    /// All entries as plain text, for copying to the clipboard.
    var text: String {
        entries.map(\.line).joined(separator: "\n")
    }
}
