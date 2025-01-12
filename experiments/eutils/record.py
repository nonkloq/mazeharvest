import csv

import os
from typing import Dict, List
import time


# from torch.utils.tensorboard import SummaryWriter


class Record:
    def __init__(
        self,
        record_name: str,
        size: int = 100,
        writer: bool = True,
        save_log: bool = True,
    ) -> None:
        self.save_logs = save_log

        # file infos
        self.__record_path = "recordlogs"
        os.makedirs(self.__record_path, exist_ok=True)

        self.timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.record_name = record_name
        self.csv_file_path = os.path.join(
            self.__record_path, f"{self.record_name}-{self.timestamp}.csv"
        )

        if not save_log:
            size = 1

        self.store: List[Dict] = [None] * size  # pyright: ignore

        self.__cur_pos = 0
        self.__max_size = size

        self.tb_writer = (
            # SummaryWriter(f"runs/{record_name}-{self.timestamp}") if writer else None
        )
        self.__c = 0

    def _save_all(self, min_lim=1):
        if not self.save_logs:
            self.__cur_pos = 0
            return

        if self.__cur_pos > min_lim:
            with open(
                self.csv_file_path, mode="a", newline="", encoding="utf-8"
            ) as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=(self.store[0].keys() if self.__cur_pos > 0 else []),
                )

                if file.tell() == 0:
                    writer.writeheader()

                writer.writerows(filter(None, self.store[: self.__cur_pos - min_lim]))

                for i, x in enumerate(range(self.__cur_pos - min_lim, self.__cur_pos)):
                    self.store[i] = self.store[x]

        self.__cur_pos = min_lim

    def append(self, rec: dict):
        if self.__cur_pos >= self.__max_size:
            self._save_all()

        if self.tb_writer:
            for k, v in rec.items():
                self.tb_writer.add_scalar(k, v, self.__c)

        self.store[self.__cur_pos] = rec

        self.__cur_pos += 1
        self.__c += 1

    def close(self):
        self._save_all(0)

        if self.tb_writer:
            self.tb_writer.close()

    def log(self):
        if self.__cur_pos > 0:  # Check if there is at least one stored record
            latest_record = self.store[self.__cur_pos - 1]
            log_message = f"{self.record_name}: " + " | ".join(
                [
                    (
                        f"{key}={value:.4f}"
                        if isinstance(value, float)
                        else f"{key}: {value}"
                    )
                    for key, value in latest_record.items()
                ]
            )

            print(log_message)
        else:
            print("No records to log.")

    def get_latest_val(self, key):
        idx = min(self.__cur_pos, self.__max_size - 1)
        return self.store[idx][key] if self.store[idx] else None

    def __del__(self):
        self.close()
