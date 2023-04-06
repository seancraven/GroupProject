"""
Mixin classes for various utilities. 
"""

try:
    import wandb
except ImportError:
    pass


class ReporterMixin:
    """
    Mixin class for reporting to wandb and stdout.

    Deals with openeing and closing wandb experiments.
    """
    @property
    def _verbosity(self):
        """ Returns the verbosity level, defaulting to 1 if not set."""
        return getattr(self, "verbosity", 1)

    def debug(self, *args, **kwargs) -> None:
        """ Print if the verbosity level is 2 or higher. """
        if self._verbosity >= 2:
            print(*args, **kwargs)

    def info(self, *args, **kwargs) -> None:
        """ Print if the verbosity level is 1 or higher. """
        if self._verbosity >= 1:
            print(*args, **kwargs)

    @staticmethod
    def wandb_init(*args, **kwargs) -> None:
        """ Try to wandb init, but don't crash if it fails."""
        try:
            wandb.init(*args, **kwargs)  # type: ignore
        except:
            pass

    @staticmethod
    def _wandb_log_no_print(*args, **kwargs) -> None:
        """ Try to wandb log, but don't crash if it fails."""
        try:
            wandb.log(*args, **kwargs)  # type: ignore
        except:
            pass

    def wandb_log(self, log_dict: dict) -> None:
        """ Log to wandb and print to stdout """
        log_string = [f"{k}: {v:.4f}" for k, v in log_dict.items()]
        log_string = "\t" + " | ".join(log_string)  # type: ignore
        self.info(log_string)
        return self._wandb_log_no_print(log_dict)

    def wandb_log_named(self, log_dict: dict, name: str) -> None:
        """ Log to wandb and print to stdout, with a name prefix."""
        named_log_dict = {f"{name} | {k}": v for k, v in log_dict.items()}
        log_strings = [f"{k}: {v:.4f}" for k, v in log_dict.items()]
        log_string = "\t" + name + " | ".join(log_strings)
        self.info(log_string)
        return self._wandb_log_no_print(named_log_dict)

    @staticmethod
    def wandb_finish():
        """ Try to wandb finish, but don't crash if it fails."""
        try:
            wandb.finish()
        except:
            pass
