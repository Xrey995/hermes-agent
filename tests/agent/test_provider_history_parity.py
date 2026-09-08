"""Provider-native replay stays private; steering survives durable history repair."""
from copy import deepcopy


import pytest

from agent.transports.chat_completions import ChatCompletionsTransport
from providers.base import ProviderProfile


def test_native_carriers_follow_only_their_owner_on_each_request():
    owner = ProviderProfile(name="native-owner")
    owner.native_reasoning_details_type = "native-owner.native_assistant"
    carrier = {"type": owner.native_reasoning_details_type, "messages": [{"text": "private"}]}
    standard = {"type": "reasoning.encrypted", "data": "opaque-signature"}
    history = [{"role": "assistant", "content": "answer", "reasoning_details": [carrier, standard]}]
    original = deepcopy(history)
    transport = ChatCompletionsTransport()
    for profile in (owner, ProviderProfile(name="other"), None, owner):
        wire = transport.build_kwargs("test", history, provider_profile=profile)["messages"]
        expected = [carrier, standard] if profile is owner else [standard]
        assert wire[0]["reasoning_details"] == expected
        assert history == original
    only_native = [{"role": "assistant", "content": "answer", "reasoning_details": [carrier]}]
    assert "reasoning_details" not in transport.convert_messages(only_native)[0]


@pytest.mark.parametrize("multimodal", [False, True])
@pytest.mark.parametrize("flush_before_steering", [False, True])
def test_steering_survives_flush_repair_and_replay(tmp_path, multimodal, flush_before_steering):
    from agent.agent_runtime_helpers import drop_thinking_only_and_merge_users, repair_message_sequence_with_cursor
    from agent.interrupt_control import append_user_steering
    from agent.session_persistence import SessionPersistenceMixin
    from agent.turn_context import build_api_messages
    from hermes_state import SessionDB
    from providers import register_provider

    profile = ProviderProfile(name="history-steering", steering_as_user_message=True)
    register_provider(profile)
    db = SessionDB(db_path=tmp_path / "history.db")
    db.create_session(session_id="steering", source="test")
    class Agent(SessionPersistenceMixin):
        pass
    agent = Agent()
    agent.__dict__.update(
        provider=profile.name, session_id="steering", _session_db=db,
        _session_db_created=True, _last_flushed_db_idx=0,
        _persist_user_message_idx=0, _persist_user_message_override="original",
        ephemeral_system_prompt=None,
        _copy_reasoning_content_for_api=lambda *_: None,
        _should_sanitize_tool_calls=lambda: False,
    )
    content = [{"type": "text", "text": "original"}] if multimodal else "original"
    messages = [{"role": "user", "content": content, **({} if multimodal else {"api_content": "context\noriginal"})}]
    try:
        if flush_before_steering:
            assert agent._flush_messages_to_session_db(messages, [])
        original = deepcopy(messages[0])
        for text in ("correction one", "correction two"):
            assert append_user_steering(agent, messages, text)
            repair_message_sequence_with_cursor(agent, messages)
            assert agent._flush_messages_to_session_db(messages, [])
        rows = db.get_messages("steering")
        assert [r["content"] for r in rows] == ["original", "correction one", "correction two"]
        assert messages[0]["content"] == original["content"]
        assert messages[0].get("api_content") == original.get("api_content")
        agent.provider = "unregistered-fallback"
        repair_message_sequence_with_cursor(agent, messages)
        assert len(messages) == len(rows)
        wire, _ = build_api_messages(
            agent, messages, current_turn_user_idx=0, ext_prefetch_cache=None,
            plugin_user_context=None, moa_config=None, active_system_prompt="stable system",
        )
        merged = drop_thinking_only_and_merge_users(wire)
        assert [m["role"] for m in merged] == ["system", "user"]
        expected = ([{"type": "text", "text": "original"},
                     {"type": "text", "text": "correction one"},
                     {"type": "text", "text": "correction two"}] if multimodal
                    else "context\noriginal\n\ncorrection one\n\ncorrection two")
        assert merged[-1]["content"] == expected
        replay = db.get_messages_as_conversation("steering")
        replay_wire, _ = build_api_messages(
            agent, replay, current_turn_user_idx=-1, ext_prefetch_cache=None,
            plugin_user_context=None, moa_config=None, active_system_prompt="stable system",
        )
        replay_merged = drop_thinking_only_and_merge_users(replay_wire)
        assert replay_merged[-1]["content"] == (
            ("original" if multimodal else "context\noriginal")
            + "\n\ncorrection one\n\ncorrection two"
        )
    finally:
        db.close()
