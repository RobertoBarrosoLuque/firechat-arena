from src.constants.configs import PROMPT_LIBRARY


def add_user_request_to_prompt(user_request: str) -> str:
    return PROMPT_LIBRARY["default_prompt"].replace("{{user_request}}", user_request)
