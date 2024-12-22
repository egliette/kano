import os
import threading
import time

import pandas as pd
import psutil


class FPSCounter:
    def __init__(
        self, start_when_init=True, fps_print_cycle=None, prefix_fps_print=""
    ):
        """
        Initializes the FPSCounter instance.

        Args:
            start_when_init (bool): Whether to start counting FPS immediately upon initialization.
            fps_print_cycle (float or None): The interval (in seconds) at which FPS will be printed.
            prefix_fps_print (str): A prefix string to include in the FPS print statement.
        """
        self.start_time = None
        self.total_frames = 0
        self.last_print_time = None
        self.fps_print_cycle = fps_print_cycle
        self.prefix_fps_print = prefix_fps_print
        if start_when_init:
            self.start()

    def start(self):
        """
        Resets the FPS counter and starts the time tracking.

        Initializes the start time and sets the total frame count to 0.
        """
        self.start_time = time.time()
        self.last_print_time = self.start_time
        self.total_frames = 0

    def update(self):
        """
        Updates the FPS counter by incrementing the frame count and optionally printing the FPS.

        If the frame count exceeds 1,000,000, the counter is reset.
        If `fps_print_cycle` is set, the FPS is printed at the specified interval.
        """
        if self.total_frames > 1_000_000:
            self.start()
        else:
            self.total_frames += 1

        if self.fps_print_cycle is not None:
            elapsed_time = time.time() - self.last_print_time
            if elapsed_time >= self.fps_print_cycle:
                print(f"{self.prefix_fps_print} FPS: {int(self.get_fps())}")
                self.start()

    def get_fps(self):
        """
        Get the current FPS (frames per second).

        Calculates FPS as the total number of frames divided by the elapsed time.

        Returns:
            float: The calculated FPS, or 0 if the counter has not started.
        """
        if self.start_time is None:
            return 0
        elapsed_time = time.time() - self.start_time
        if elapsed_time == 0:
            return 0
        fps = self.total_frames / elapsed_time
        return fps

    def keep_target_fps(self, target_fps):
        """
        Ensure the FPS stays below or at the target FPS by adjusting the sleep time.

        Args:
            target_fps (float): The target FPS to maintain.
        """
        current_fps = self.get_fps()
        if current_fps > target_fps:
            sleep_time = self.total_frames / target_fps - (
                time.time() - self.start_time
            )
            if sleep_time > 0:
                time.sleep(sleep_time)


class ResourceProfiler:
    def __init__(
        self, interval_seconds, pid=None, csv_path=None, csv_minutes=5
    ):
        """
        Initializes the ResourceProfiler instance.

        Args:
            interval_seconds (float): The interval in seconds between each resource update.
            pid (int or None): The process ID to monitor, or None to monitor the current process.
            csv_path (str or None): Path to a CSV file to save resource data, or None to skip saving.
            csv_minutes (int): The number of minutes of data to retain in the CSV file.
        """
        self.interval_seconds = interval_seconds
        self.last_update_time = time.time()
        if pid:
            self.current_process = psutil.Process(pid)
            self.pid = pid
        else:
            self.current_process = psutil.Process()
            self.pid = os.getpid()
        self.csv_path = csv_path
        self.csv_minutes = csv_minutes
        self.time_format = "%Y-%m-%d %H:%M:%S"

    def update_csv(self, current_time, cpu_percent, ram_mib):
        """
        Update the CSV file with the current resource usage.

        Args:
            current_time (float): The current timestamp.
            cpu_percent (float): The current CPU usage as a percentage.
            ram_mib (float): The current RAM usage in MiB.
        """
        if os.path.isfile(self.csv_path):
            df = pd.read_csv(self.csv_path)
            df = df[df["time"] > current_time - self.csv_minutes * 60]
        else:
            df = pd.DataFrame(columns=["time", "cpu_percent", "ram_mib"])

        new_line = {
            "time": current_time,
            "cpu_percent": cpu_percent,
            "ram_mib": ram_mib,
        }

        df.loc[len(df)] = new_line
        df.to_csv(self.csv_path, index=False, header=True)

    def get_current_info(self):
        """
        Get the current process resource usage (CPU and RAM).

        Returns:
            tuple: A tuple containing the current timestamp, CPU usage percentage, and RAM usage in MiB.
        """
        current_time = time.time()
        cpu_percent = self.current_process.cpu_percent()
        ram_mib = self.current_process.memory_info().rss / 1024**2
        return current_time, cpu_percent, ram_mib

    def update(self):
        """
        Update the resource usage information and optionally print it.

        If the interval has elapsed, the current CPU and RAM usage will be printed.
        If a CSV path is specified, the data will be written to the CSV file.
        """
        current_time, cpu_percent, ram_mib = self.get_current_info()
        if current_time - self.last_update_time >= self.interval_seconds:
            print(
                f"PID: {self.pid} - CPU Usage: {cpu_percent}% - total RAM: {ram_mib:.2f} MiB"
            )
            self.last_update_time = current_time
        if self.csv_path:
            self.update_csv(current_time, cpu_percent, ram_mib)

    def _profiling(self):
        """
        Continuously profiles the resources at the specified interval.

        This method runs in a separate thread and continuously updates the resource usage.
        """
        while True:
            current_time, cpu_percent, ram_mib = self.get_current_info()
            if current_time - self.last_update_time >= self.interval_seconds:
                print(
                    f"PID: {self.pid} - CPU Usage: {cpu_percent}% - total RAM: {ram_mib:.2f} MiB"
                )
                self.last_update_time = current_time
            if self.csv_path:
                self.update_csv(current_time, cpu_percent, ram_mib)
            time.sleep(self.interval_seconds)

    def start_profiling_thread(self):
        """
        Start a separate thread for continuous resource profiling.

        The thread runs the `_profiling` method to collect and display resource usage.
        """
        thread = threading.Thread(target=self._profiling)
        thread.start()


class FPSProfiler(ResourceProfiler):
    def __init__(
        self,
        interval_seconds,
        pid=None,
        csv_path=None,
        csv_minutes=5,
        target_fps=None,
    ):
        """
        Initializes the FPSProfiler instance, extending ResourceProfiler to track FPS.

        Args:
            interval_seconds (float): The interval in seconds between each resource update.
            pid (int or None): The process ID to monitor, or None to monitor the current process.
            csv_path (str or None): Path to a CSV file to save resource and FPS data.
            csv_minutes (int): The number of minutes of data to retain in the CSV file.
            target_fps (float or None): The target FPS to maintain.
        """
        super().__init__(interval_seconds, pid, csv_path, csv_minutes)
        self.fps_counter = FPSCounter()
        self.target_fps = target_fps

    def update_csv(self, current_time, cpu_percent, ram_mib, fps):
        """
        Update the CSV file with the current resource usage and FPS.

        Args:
            current_time (float): The current timestamp.
            cpu_percent (float): The current CPU usage as a percentage.
            ram_mib (float): The current RAM usage in MiB.
            fps (int): The current frames per second.
        """
        if os.path.isfile(self.csv_path):
            df = pd.read_csv(self.csv_path)
            df = df[df["time"] > current_time - self.csv_minutes * 60]
        else:
            df = pd.DataFrame(
                columns=["time", "cpu_percent", "ram_mib", "fps"]
            )
        new_line = {
            "time": current_time,
            "cpu_percent": cpu_percent,
            "ram_mib": ram_mib,
            "fps": fps,
        }
        df.loc[len(df)] = new_line
        df.to_csv(self.csv_path, index=False, header=True)

    def get_current_info(self):
        """
        Get the current resource usage and FPS.

        Returns:
            tuple: A tuple containing the current timestamp, CPU usage percentage, RAM usage in MiB, and FPS.
        """
        current_time, cpu_percent, ram_mib = super().get_current_info()
        self.fps_counter.update()
        fps = int(self.fps_counter.get_fps())
        return current_time, cpu_percent, ram_mib, fps

    def update(self):
        """
        Update the resource usage and FPS information, printing and saving to CSV if applicable.

        If the interval has elapsed, the current CPU, RAM, and FPS usage will be printed.
        If a CSV path is specified, the data will be written to the CSV file.
        If a target FPS is defined, attempts to maintain the target FPS.

        Returns:
            tuple: A tuple containing CPU usage percentage, RAM usage in MiB, and FPS.
        """
        current_time, cpu_percent, ram_mib, fps = self.get_current_info()
        if current_time - self.last_update_time >= self.interval_seconds:
            print(
                f"PID: {self.pid} - CPU Usage: {cpu_percent}% - total RAM: {ram_mib:.2f} MiB - FPS: {fps}"
            )
            self.last_update_time = current_time
        if self.csv_path:
            self.update_csv(current_time, cpu_percent, ram_mib, fps)
        if self.target_fps:
            self.fps_counter.keep_target_fps(self.target_fps)
        return cpu_percent, ram_mib, fps
