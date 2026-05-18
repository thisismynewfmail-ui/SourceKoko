import sys, json, pyttsx3
cfg = json.loads(sys.argv[1])
engine = pyttsx3.init()
vid = cfg.get("voice", "")
if vid:
    engine.setProperty("voice", vid)
engine.setProperty("rate", int(cfg.get("rate", 170)))
engine.setProperty("volume", min(1.0, max(0.0, int(cfg.get("volume", 100)) / 100.0)))
outfile = cfg["outfile"]
engine.save_to_file(cfg["text"], outfile)
engine.runAndWait()
print("OK")
