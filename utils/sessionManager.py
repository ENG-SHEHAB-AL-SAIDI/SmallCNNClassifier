import logging
import csv
import shutil
from pathlib import Path
from datetime import datetime
import torch

class TrainingSessionManager:
    def __init__(
        self,
        baseDir: str,
        configFilePath: str,
        resumeFromSessionId: str | None = None,  # optional old session to resume weights
        isLoggingEnabled: bool = True,
        logLevel: int = logging.INFO
    ):
        self.baseDir = Path(baseDir)
        self.baseDir.mkdir(exist_ok=True)

           # -------------------------
        # 1️⃣ Determine source session
        # -------------------------
        self.sourceSessionDir = None
        if resumeFromSessionId:
            if resumeFromSessionId.lower() == "last":
                sessions = sorted([d for d in self.baseDir.iterdir() if d.is_dir()])
                if sessions:
                    self.sourceSessionDir = sessions[-1]  # latest session
            else:
                self.sourceSessionDir = self.baseDir / resumeFromSessionId
                if not self.sourceSessionDir.exists():
                    raise FileNotFoundError(f"Session {resumeFromSessionId} does not exist")
          

        # -------------------------
        # 2️⃣ New session for logs, metrics, and checkpoints
        # -------------------------
        self.sessionDir = self._createNewSessionDir()
        self.sessionId = self.sessionDir.name
        self.checkpointDir = self.sessionDir / "checkpoints"
        self.checkpointDir.mkdir(exist_ok=True)

        # -------------------------
        # 3️⃣ Save config snapshot
        # -------------------------
        self._saveConfigSnapshot(configFilePath)

        # -------------------------
        # 4️⃣ Setup logger
        # -------------------------
        self.logger = self._createLogger(isLoggingEnabled, logLevel)
        self.metricsPath = self.sessionDir / "metrics.csv"

    # -------------------------
    # New session folder
    # -------------------------
    def _createNewSessionDir(self) -> Path:
        sessionId = datetime.now().strftime("%Y%m%d_%H%M%S")
        sessionDir = self.baseDir / sessionId
        sessionDir.mkdir(parents=True, exist_ok=True)
        return sessionDir

    # -------------------------
    # Save config snapshot
    # -------------------------
    def _saveConfigSnapshot(self, configFilePath: str) -> None:
        shutil.copy(
            Path(configFilePath),
            self.sessionDir / "config_snapshot.py"
        )

    # -------------------------
    # Logger
    # -------------------------
    def _createLogger(self, isLoggingEnabled: bool, logLevel: int) -> logging.Logger:
        loggerName = f"training_{self.sessionId}"
        logger = logging.getLogger(loggerName)
        logger.setLevel(logLevel)
        logger.propagate = False

        if logger.handlers:
            return logger

        if not isLoggingEnabled:
            logger.addHandler(logging.NullHandler())
            return logger

        handler = logging.FileHandler(
            self.sessionDir / "training.log",
            encoding="utf-8"
        )
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | epoch=%(epoch)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    # -------------------------
    # Metrics
    # -------------------------
    def logMetric(self, epoch: int, **metrics: float) -> None:
        isNewFile = not self.metricsPath.exists()
        with open(self.metricsPath, "a", newline="") as f:
            writer = csv.writer(f)
            if isNewFile:
                writer.writerow(["epoch", *metrics.keys()])
            writer.writerow([epoch, *metrics.values()])

    # -------------------------
    # Checkpoints
    # -------------------------
    def saveCheckpoint(self, model, optimizer, epoch: int) -> None:
        torch.save(
            {
                "epoch": epoch,
                "modelState": model.state_dict(),
                "optimizerState": optimizer.state_dict()
            },
            self.checkpointDir / f"epoch_{epoch:03d}.pth"
        )

    def loadWeightsFromSource(self, model, optimizer) -> int:
        """
        Load model & optimizer weights from source session (if provided)
        Logs which session, epoch, and last losses if available.
        """
        if not self.sourceSessionDir:
            self.logger.info("No source session provided, training from scratch.", extra={"epoch": 0})
            return 0  # start from scratch

        checkpointDir = self.sourceSessionDir / "checkpoints"
        checkpoints = sorted(checkpointDir.glob("epoch_*.pth"))
        if not checkpoints:
            self.logger.info(
                f"No checkpoints found in session {self.sourceSessionDir.name}, training from scratch.",
                extra={"epoch": 0}
            )
            return 0

        # Load the last checkpoint
        lastCheckpoint = checkpoints[-1]
        data = torch.load(lastCheckpoint, map_location="cpu")
        model.load_state_dict(data["modelState"])
        optimizer.load_state_dict(data["optimizerState"])

        # Try to log last losses if stored in checkpoint (optional)
        last_epoch = data.get("epoch", None)
        train_loss = data.get("train_loss", None)
        train_acc = data.get("train_acc", None)
        test_loss = data.get("test_loss", None)
        test_acc = data.get("test_acc", None)

        msg = f"Resuming from session '{self.sourceSessionDir.name}', checkpoint '{lastCheckpoint.name}'"
        if train_loss is not None and train_acc is not None and test_loss is not None and test_acc is not None:
            msg += f", epoch={last_epoch}, train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, test_loss={test_loss:.4f}, test_acc={test_acc:.4f}"
        else:
            msg += f", epoch={last_epoch if last_epoch is not None else 'unknown'}"

        self.logger.info(msg, extra={"epoch": last_epoch if last_epoch is not None else 0})
        print(msg, extra={"epoch": last_epoch if last_epoch is not None else 0})

        return last_epoch + 1 if last_epoch is not None else 0


    # -------------------------
    # Final model
    # -------------------------
    def saveFinalModel(self, model) -> None:
        torch.save(
            model.state_dict(),
            self.sessionDir / "model_final.pth"
        )
