"""Reuse the bounded pi05 transport; Metis has its own joint7 handshake and request IDs."""

from ..robotwin_pi05.client import PolicyClient as BoundedClient
from .contract import decode_actions, require_same_contract


class PolicyClient(BoundedClient):
    contract_key = "metis_contract"
    check_contract = staticmethod(require_same_contract)
    check_actions = staticmethod(decode_actions)

    def __init__(self, contract, *, port=8006, timeout_s=1.5):
        self.sequence = 0
        super().__init__(contract, port=port, timeout_s=timeout_s)

    def infer(self, observation):
        self.sequence += 1
        result = super().infer(dict(observation, request_id=self.sequence))
        if result.get("request_id") != self.sequence:
            self.close()
            raise ValueError("Metis response does not match the observation request")
        return result
