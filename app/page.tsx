import case001 from "../data/cases/case-001.json";
import case002 from "../data/cases/case-002.json";
import { InterrogationClient } from "./ui/interrogation-client";

export default function Home() {
  return (
    <main className="app-shell poly-app-shell">
      <header className="poly-titlebar">
        <div className="poly-wordmark">
          <span className="micro-label">J-Lens field terminal / incident 001</span>
          <h1>Telepathic <span>Detective</span></h1>
        </div>
        <div className="poly-system-line" aria-label="System status">
          <span><i aria-hidden="true" /> Lens online</span>
          <span>Qwen 3.5 · 4B</span>
          <span>Colony time 03:17</span>
        </div>
      </header>

      <InterrogationClient
        openingBrief={case001.opening_brief}
        title={case001.title}
        turnLimit={20}
        suspects={[
          {
            id: "ilya",
            name: case001.suspect_name,
            role: case001.suspect_role,
            blurb: "Chief Habitat Engineer. Controlled, technical, resentful of command euphemisms.",
            sampleQuestions: [
              "Where exactly were you when the breach alarm sounded?",
              "What did Lena's audit threaten to expose?",
              "Why did the corridor timing change before the breach?",
              "What do you remember about the station's last sunrise?"
            ]
          },
          {
            id: "mira",
            name: case002.suspect_name,
            role: case002.suspect_role,
            blurb: "Security Chief. Procedural, defensive, and certain the records clear her.",
            sampleQuestions: [
              "Where exactly were you when the breach alarm sounded?",
              "What did Lena's audit threaten to expose?",
              "Why did the corridor timing change before the breach?",
              "What do you remember about the station's last sunrise?"
            ]
          }
        ]}
        accusationOptions={{
          suspect: [
            { value: "ilya", label: "Ilya Soren" },
            { value: "mira", label: "Captain Mira Tal" }
          ],
          motive: case001.accusation_options.motive.map((value) => ({ value, label: value })),
          method: case001.accusation_options.method.map((value) => ({ value, label: value }))
        }}
      />
    </main>
  );
}
