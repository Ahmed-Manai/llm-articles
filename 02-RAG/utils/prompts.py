"""Step 4 of RAG: prompt templates that turn variables into chat messages."""
import re
from typing import Dict, List


class RolePrompt:
    """A template like "Hello {name}" tied to a chat role (system, user or assistant)."""

    def __init__(self, template: str, role: str):
        self.template = template
        self.role = role

    def get_input_variables(self) -> List[str]:
        return re.findall(r"\{([^}]+)\}", self.template)

    def create_message(self, **kwargs) -> Dict[str, str]:
        missing = set(self.get_input_variables()) - kwargs.keys()
        if missing:
            raise ValueError(f"Missing prompt variables: {sorted(missing)}")
        return {"role": self.role, "content": self.template.format(**kwargs)}


class SystemRolePrompt(RolePrompt):
    def __init__(self, template: str):
        super().__init__(template, "system")


class UserRolePrompt(RolePrompt):
    def __init__(self, template: str):
        super().__init__(template, "user")


class AssistantRolePrompt(RolePrompt):
    def __init__(self, template: str):
        super().__init__(template, "assistant")


if __name__ == "__main__":
    prompt = SystemRolePrompt("You are an expert in {expertise}.")
    print(prompt.get_input_variables())
    print(prompt.create_message(expertise="Python"))
