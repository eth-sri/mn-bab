from typing import Optional


class VerificationTimeoutException(RuntimeError):
    """When verification times out."""

    def __init__(
        self,
        message: Optional[str] = None,
        lb: Optional[float] = None,
        ub: Optional[float] = None,
    ):
        if message is None:
            message = "Verification timed out."
        super().__init__(message)

        self.best_lb = lb
        self.best_ub = ub
