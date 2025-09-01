# core/sequence.py (新增)
class Sequence:
    def __init__(self, seq_id: int, prompt: str):
        self.seq_id = seq_id
        self.prompt = prompt
        self.input_ids = []
        self.output_ids = []
        self.ended = False
        self.prefill_done = False
        self.next_token = None

    def add_output_token(self, token_id: int):
        self.output_ids.append(token_id)

    def get_last_token(self):
        return self.output_ids[-1] if self.output_ids else None

    def get_full_input(self):
        return self.input_ids + self.output_ids