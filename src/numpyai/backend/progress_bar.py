"""Progress Bar class."""

from __future__ import annotations
import time
import threading
import shutil
from typing import Iterable, ClassVar, Optional, Sized

class ProgressBar:
    """Decorates an iterable with a progress bar that is printed to the console."""

    DEFAULT_ANIMATION: ClassVar[list[str]] = ['-', '\\', '|', '/']

    def __init__(self, iterable: Iterable, desc: str = "Processing", total: Optional[int] = None, 
                 bars: Optional[int] = None, display_interval: float = 0.01, smoothing: float = 0.1,
                 animation_tick_multiplier: int = 10, animation: Optional[list[str]] = None, 
                 newline_close: bool = True) -> None:
        """
        Creates a new progress bar.

        Parameters
        ----------
        iterable : Iterable
            The iterable to wrap
        desc : str, optional
            Description to display with the progress bar, by default 'Processing'
        total : int | None, optional
            The total number of items in the iterable, by default `len(iterable)`
        bars : int | None, optional
            Width of the progress bar in characters, by default fills the console width
        display_interval : float, optional
            Minimum time in seconds between display updates, by default 0.01
        smoothing : float, optional
            Smoothing factor for EMA estimation, by default 0.1
        animation_tick_multiplier : int, optional
            How many display updates must pass before the animation frame advances, by default 10
        animation : list[str] | None, optional
            List of characters for the progress bar's animation, by default None
        newline_close : bool, optional
            Whether to print a newline when the progress bar is closed, by default True
        """
        self.iterator = iter(iterable)        
        self.desc = desc
        self.bars = bars
        self.smoothing = smoothing
        self.display_interval = display_interval
        self.animation_tick_multiplier = max(1, int(animation_tick_multiplier))
        self.animation = animation or self.DEFAULT_ANIMATION
        self.newline_close = newline_close

        if total:
            self.total = total
        elif isinstance(iterable, Sized):
            self.total = len(iterable)
        else:
            self.total = None
        self.total_width = len(str(self.total))

        # Display threading components
        self._stop_event = threading.Event()
        self._thread = None

        # State variables
        self.n = 0
        self.start_time = 0
        self.last_update_time = 0
        self.last_estimate_time = 0
        self.estimate = float('inf')
        self.smoothed_rate = 0
        self.animation_index = 0
        self.first_update = True

    def _painter_thread(self) -> None:
        """Background thread's loop to continuously update the progress bar display."""
        tick_counter = 0
        while not self._stop_event.wait(self.display_interval):
            if tick_counter % self.animation_tick_multiplier == 0:
                self.animation_index = (self.animation_index + 1) % len(self.animation)
            self.display()
            tick_counter += 1

    def __iter__(self) -> ProgressBar:
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.last_estimate_time = self.start_time
        if not self._thread:
            self._thread = threading.Thread(target=self._painter_thread, daemon=True)
            self._thread.start()
        return self
    
    def __next__(self):
        try:
            item = next(self.iterator)
            self.update()
            return item
        except StopIteration:
            self.close()
            raise

    def __len__(self) -> int:
        return self.total or 0
    
    def __enter__(self) -> ProgressBar:
        return self.__iter__()
    
    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def update(self, change: int = 1) -> None:
        """Updates the progress bar's state and redraws if necessary."""
        if self.total and self.n >= self.total:
            return
        
        self.n += change
        current_time = time.time()

        # Ignore first update for time estimation
        if self.first_update:
            self.last_update_time = current_time
            self.first_update = False
            return
        
        # Update time estimate using EMA
        time_delta = current_time - self.last_update_time
        if time_delta > 0:
            instant_rate = change / time_delta
            if self.smoothed_rate > 0:
                self.smoothed_rate = (self.smoothing * instant_rate) + ((1 - self.smoothing) * self.smoothed_rate)
            else:
                self.smoothed_rate = instant_rate

            if self.total and self.smoothed_rate > 0:
                self.estimate = (self.total - self.n) / self.smoothed_rate
                self.last_estimate_time = current_time
            
        self.last_update_time = current_time


    def display(self, include_estimate: bool = True, end: str = '\r', pad: bool = True, animation_index: Optional[int] = None) -> None:
        """Prints the current progress bar and information."""
        terminal_width = shutil.get_terminal_size().columns
        current_time = time.time()
        elapsed_str = f'Elapsed: {current_time - self.start_time:.1f}s'
        animation_str = f'{self.animation[animation_index if animation_index is not None else self.animation_index]:^5s}'

        if self.total:
            percentage = self.n / self.total if self.total > 0 else 1
            percentage_str = f'{percentage:>6.1%}'

            if self.estimate == float('inf'):
                estimate_str = ' - Estimate time: --.-s'
            else:
                estimate_str = f' - Estimate time: {max(0, self.estimate - current_time + self.last_estimate_time):.1f}s'

            fixed_width = len(animation_str) + len(self.desc) + len(percentage_str) + 2 * self.total_width + len(elapsed_str) + len(estimate_str) + 11

            max_bar_width = terminal_width - fixed_width
            bar_width = min(self.bars, max_bar_width) if self.bars else max(5, max_bar_width)
        
            # Calculates amount of filled bars and uses it to construct progress bars
            bars_filled = int(percentage * bar_width)
            if self.n < self.total:
                progress = '[' + '=' * bars_filled + '>' + '.' * (bar_width - bars_filled - 1) + ']'
            else:
                progress = '[' + '=' * bar_width + ']'

            output = (
                f'{animation_str}{self.desc} - {percentage_str} {progress} '
                f'{self.n:{self.total_width}d}/{self.total} - {elapsed_str}{estimate_str if include_estimate else ""}'
            )
        else:
            rate_str = f' [{self.smoothed_rate:.1f} items/s]' if include_estimate and self.smoothed_rate > 0 else ''
            output = f'{animation_str}{self.desc} - {self.n} items processed - {elapsed_str}{rate_str}'
        if pad:
            output = output.ljust(terminal_width)
        print(output[:terminal_width], end=end, flush=True)

    def close(self):
        """Disables the progress bar and prints the final display."""
        if self._thread and self._thread.is_alive():
            self._stop_event.set()
            self._thread.join()
            
        print(' ' * shutil.get_terminal_size().columns, end='\r', flush=True)
        self.display(include_estimate=False, end='\n' if self.newline_close else '', pad=False, animation_index=0)
