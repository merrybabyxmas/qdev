from abc import ABC, abstractmethod


class BrokerInterface(ABC):
    @abstractmethod
    def connect(self):
        pass

    @abstractmethod
    def disconnect(self):
        pass

    @abstractmethod
    def heartbeat(self):
        """Return a lightweight connectivity snapshot."""
        pass

    @abstractmethod
    def get_account(self):
        pass

    @abstractmethod
    def get_positions(self):
        pass

    @abstractmethod
    def sync_state(self):
        """Reconcile account, positions, open orders, and fills from broker state."""
        pass

    @abstractmethod
    def get_latest_price(self, symbol):
        pass

    @abstractmethod
    def place_order(self, order):
        pass

    @abstractmethod
    def cancel_order(self, order_id):
        pass

    @abstractmethod
    def get_open_orders(self):
        pass

    @abstractmethod
    def get_fills(self):
        pass
