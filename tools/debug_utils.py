"""
Debug utilities for controlling print statements in model training.
"""

class DebugPrinter:
    """
    A wrapper for print statements that allows conditional printing based on:
    - Current epoch (only print after a certain epoch)
    - Sample count (only print for last n samples per epoch)
    """
    
    def __init__(self, start_epoch=0, last_n_samples=5, enabled=True):
        """
        Args:
            start_epoch (int): Only print after this epoch (inclusive)
            last_n_samples (int): Only print for last n samples per epoch
            enabled (bool): Global enable/disable switch
        """
        self.start_epoch = start_epoch
        self.last_n_samples = last_n_samples
        self.enabled = enabled
        self.mode = 'train'  # or 'eval'
        
        # State tracking
        self.current_epoch = 0
        self.current_sample = 0
        self.total_samples_per_epoch = None

    def set_type(self, mode):
        """Set the mode type: 'train' or 'eval'"""
        self.mode = mode
        
    def set_epoch(self, epoch):
        """Set the current epoch"""
        self.current_epoch = epoch
        self.current_sample = 0  # Reset sample counter for new epoch
        
    def set_total_samples(self, total_samples):
        """Set total samples per epoch (needed to determine last n samples)"""
        self.total_samples_per_epoch = total_samples
        
    def set_sample(self, sample_idx):
        """Set the current sample index within the epoch"""
        self.current_sample = sample_idx
        
    def increment_sample(self):
        """Increment the sample counter"""
        self.current_sample += 1
        
    def should_print(self):
        """Check if we should print based on current conditions"""
        # print(f"DebugPrinter: mode={self.mode}, enabled={self.enabled}, current_epoch={self.current_epoch}, start_epoch={self.start_epoch}, current_sample={self.current_sample}, total_samples_per_epoch={self.total_samples_per_epoch}, last_n_samples={self.last_n_samples}")
        
        if not self.mode == 'train':    
            return False

        if not self.enabled:
            return False
            
        # Check epoch condition
        if self.current_epoch < self.start_epoch:
            return False
            
        # Check sample condition (last n samples)
        if self.total_samples_per_epoch is not None:
            samples_from_end = self.total_samples_per_epoch - self.current_sample
            if samples_from_end > self.last_n_samples:
                return False
                
        return True
    
    def print(self, *args, **kwargs):
        """Conditional print function"""
        if self.should_print():
            print(f"[{self.mode}: Epoch {self.current_epoch}, Sample {self.current_sample}]", *args, **kwargs)

    # Function variant to print the result of a passed function call
    def print_of_function(self, func, *args, **kwargs):
        """Conditional print function for content"""
        if self.should_print():
            content = func(*args, **kwargs)
            print(f"[{self.mode}: Epoch {self.current_epoch}, Sample {self.current_sample} ] {content}")

    def print_of_function_force_print(self, func, *args, force_print = True, **kwargs):
        """Conditional print function for content"""
        if force_print:
            content = func(*args, **kwargs)
            print(f"[{self.mode}: Epoch {self.current_epoch}, Sample {self.current_sample} (forced)] {content}")
    
    def configure(self, start_epoch=None, last_n_samples=None, enabled=None):
        """Update configuration"""
        if start_epoch is not None:
            self.start_epoch = start_epoch
        if last_n_samples is not None:
            self.last_n_samples = last_n_samples
        if enabled is not None:
            self.enabled = enabled


# Global instance that can be used across the project
# Configuration examples:
# debug_printer.configure(start_epoch=25, last_n_samples=5, enabled=True)  # Print only after epoch 25, last 5 samples
# debug_printer.configure(start_epoch=0, last_n_samples=10, enabled=True)   # Print from beginning, last 10 samples
# debug_printer.configure(enabled=False)                                    # Disable all debug printing
debug_printer = DebugPrinter(start_epoch=0, last_n_samples=10, enabled=True)
