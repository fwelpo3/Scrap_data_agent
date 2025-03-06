// server.js
const express = require("express");
const fs = require("fs");
const path = require("path");

const app = express();
const PORT = 3000;

app.use(express.json());

// <<< Hier statische Dateien freigeben
// __dirname zeigt auf das Verzeichnis, in dem server.js liegt
app.use(express.static(__dirname));

// Beispielrouten zum Anlegen/Auslesen der Profile:
app.get("/api/profiles", (req, res) => {
  try {
    const data = fs.readFileSync(path.join(__dirname, "db.json"), "utf8");
    const profiles = JSON.parse(data);
    res.json(profiles);
  } catch (err) {
    console.error("Fehler beim Lesen der Profile:", err);
    res.status(500).json({ error: "Fehler beim Lesen der Profile" });
  }
});

app.post("/api/profile", (req, res) => {
  try {
    const filePath = path.join(__dirname, "db.json");
    const fileData = fs.readFileSync(filePath, "utf8");
    let profiles = JSON.parse(fileData);
    profiles.push(req.body);
    fs.writeFileSync(filePath, JSON.stringify(profiles, null, 2));
    res.status(201).json({ message: "Profil erfolgreich gespeichert" });
  } catch (error) {
    console.error("Fehler beim Speichern:", error);
    res.status(500).json({ error: "Fehler beim Speichern des Profils" });
  }
});

app.listen(PORT, () => {
  console.log(`Server läuft auf http://localhost:${PORT}`);
});
