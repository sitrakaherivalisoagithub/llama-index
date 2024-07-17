from typing import Dict, List, Optional
from llama_index.core.llms import ChatMessage
from llama_index.core.storage.chat_store import SimpleChatStore


class Session:
    def __init__(self, user_id: str, chat_store: SimpleChatStore, persist_path: str = "chat_store.json"):
        self.user_id = user_id
        self.chat_store = chat_store
        self.persist_path = persist_path
        self.initialize_user()

    def initialize_user(self) -> None:
        """Initialize user by checking if user exists in the store."""
        if self.user_id not in self.chat_store.get_keys():
            self.chat_store.set_messages(self.user_id, [])
            self.chat_store.persist(self.persist_path)

    def add_message(self, message: ChatMessage) -> None:
        self.chat_store.add_message(self.user_id, message)
        self.chat_store.persist(self.persist_path)

    def set_messages(self, messages: List[ChatMessage]) -> None:

        self.chat_store.set_messages(self.user_id, messages)
        self.chat_store.persist(self.persist_path)

    def get_messages(self) -> List[ChatMessage]:
        return self.chat_store.get_messages(self.user_id)

    def delete_message(self, idx: int) -> Optional[ChatMessage]:
        message = self.chat_store.delete_message(self.user_id, idx)
        self.chat_store.persist(self.persist_path)
        return message

    def delete_last_message(self) -> Optional[ChatMessage]:
        message = self.chat_store.delete_last_message(self.user_id)
        self.chat_store.persist(self.persist_path)
        return message

    def delete_all_messages(self) -> Optional[List[ChatMessage]]:
        messages = self.chat_store.delete_messages(self.user_id)
        self.chat_store.persist(self.persist_path)
        return messages

"""
 Exemple d'utilisation
persist_path = "chat_store.json"
chat_store = SimpleChatStore.from_persist_path(persist_path)

# Créer une nouvelle session pour un utilisateur avec l'ID 'user123'
session = Session(user_id='user123', chat_store=chat_store, persist_path=persist_path)

# Ajouter un message pour l'utilisateur
message = ChatMessage(role="user", content="Hello, world!")
session.add_message(message)

# Récupérer tous les messages de l'utilisateur
print(session.get_messages())

# Supprimer le dernier message de l'utilisateur
session.delete_last_message()

# Supprimer tous les messages de l'utilisateur
session.delete_all_messages()
https://storage.googleapis.com/github-repo/use-cases/sheet-music/24ItalianSongs.pdf
"""

