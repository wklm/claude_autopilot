"""Unit tests for retry strategy with exponential backoff."""

from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from claude_code_agent_farm.models.retry_strategy import (
    RetryStrategy,
    UsageLimitRetryInfo,
)


class TestRetryStrategy:
    """Test retry strategy functionality."""
    
    def test_initial_delay(self):
        """Test initial delay calculation."""
        strategy = RetryStrategy(initial_delay_seconds=60)
        delay = strategy.calculate_next_delay()
        
        # Should be close to initial delay (with some jitter)
        assert 54 <= delay <= 66  # 60 ± 10%
    
    def test_exponential_backoff(self):
        """Test exponential backoff increases."""
        strategy = RetryStrategy(
            initial_delay_seconds=60,
            backoff_factor=2.0,
            jitter_factor=0.0  # No jitter for predictable test
        )
        
        # First attempt
        assert strategy.calculate_next_delay() == 60
        
        # Record attempts and check backoff
        strategy.current_attempt = 1
        assert strategy.calculate_next_delay() == 120  # 60 * 2^1
        
        strategy.current_attempt = 2
        assert strategy.calculate_next_delay() == 240  # 60 * 2^2
        
        strategy.current_attempt = 3
        assert strategy.calculate_next_delay() == 480  # 60 * 2^3
    
    def test_max_delay_cap(self):
        """Test that delay is capped at max."""
        strategy = RetryStrategy(
            initial_delay_seconds=60,
            max_delay_seconds=300,
            backoff_factor=2.0,
            jitter_factor=0.0
        )
        
        # High attempt count
        strategy.current_attempt = 10
        delay = strategy.calculate_next_delay()
        
        # Should be capped at max
        assert delay == 300
    
    def test_jitter_adds_randomness(self):
        """Test that jitter adds randomness."""
        strategy = RetryStrategy(
            initial_delay_seconds=100,
            jitter_factor=0.2
        )
        
        # Get multiple delays
        delays = [strategy.calculate_next_delay() for _ in range(10)]
        
        # Should have variation
        assert len(set(delays)) > 1
        
        # All should be within jitter range
        for delay in delays:
            assert 80 <= delay <= 120  # 100 ± 20%
    
    def test_should_retry_limits(self):
        """Test retry attempt limits."""
        strategy = RetryStrategy(max_retry_attempts=3)
        
        # Should allow retries initially
        assert strategy.should_retry() == True
        
        # Record attempts up to limit
        for i in range(3):
            strategy.record_retry_attempt()
        
        # Should not allow after limit
        assert strategy.should_retry() == False
    
    def test_retry_window_reset(self):
        """Test retry window resets old attempts."""
        strategy = RetryStrategy(
            max_retry_attempts=3,
            retry_window_hours=1
        )
        
        # Add old attempts
        old_time = datetime.now() - timedelta(hours=2)
        strategy.retry_history = [old_time, old_time]
        strategy.current_attempt = 2
        
        # Should reset and allow retry
        assert strategy.should_retry() == True
        assert len(strategy.retry_history) == 0
        assert strategy.current_attempt == 0
    
    def test_record_retry_attempt(self):
        """Test recording retry attempts."""
        strategy = RetryStrategy()
        
        # Record attempts
        strategy.record_retry_attempt()
        strategy.record_retry_attempt()
        
        assert strategy.current_attempt == 2
        assert len(strategy.retry_history) == 2
        
        # Check timestamps are recent
        for timestamp in strategy.retry_history:
            assert (datetime.now() - timestamp).total_seconds() < 1
    
    def test_reset_clears_state(self):
        """Test reset clears all state."""
        strategy = RetryStrategy()
        
        # Add some state
        strategy.current_attempt = 5
        strategy.last_retry_time = datetime.now()
        strategy.retry_history = [datetime.now(), datetime.now()]
        
        # Reset
        strategy.reset()
        
        assert strategy.current_attempt == 0
        assert strategy.last_retry_time is None
        assert len(strategy.retry_history) == 0
    
    def test_get_next_retry_time(self):
        """Test calculating next retry time."""
        strategy = RetryStrategy(
            initial_delay_seconds=60,
            jitter_factor=0.0
        )
        
        with patch('claude_code_agent_farm.models.retry_strategy.datetime') as mock_dt:
            now = datetime(2024, 1, 1, 12, 0, 0)
            mock_dt.now.return_value = now
            
            retry_time = strategy.get_next_retry_time()
            
            expected = now + timedelta(seconds=60)
            assert retry_time == expected


class TestUsageLimitRetryInfo:
    """Test usage limit retry info functionality."""
    
    def test_parsed_time_used_when_valid(self):
        """Test parsed time is used when valid."""
        info = UsageLimitRetryInfo(message="Limit reached")
        
        future_time = datetime.now() + timedelta(hours=1)
        info.set_parsed_time(future_time)
        
        assert info.effective_retry_time == future_time
        assert not info.use_fallback
    
    def test_fallback_when_no_parsed_time(self):
        """Test fallback when parsing fails."""
        info = UsageLimitRetryInfo(message="Limit reached")
        info.set_parsed_time(None)
        
        assert info.use_fallback
        assert info.fallback_reason == "Could not parse retry time from message"
        assert info.calculated_retry_time is not None
    
    def test_fallback_when_parsed_time_in_past(self):
        """Test fallback when parsed time is in past."""
        info = UsageLimitRetryInfo(message="Limit reached")
        
        past_time = datetime.now() - timedelta(hours=1)
        info.set_parsed_time(past_time)
        
        assert info.use_fallback
        assert info.fallback_reason == "Parsed time is in the past"
        assert info.calculated_retry_time > datetime.now()
    
    def test_effective_retry_time_hierarchy(self):
        """Test effective retry time selection hierarchy."""
        info = UsageLimitRetryInfo(message="Limit reached")
        
        # Case 1: Valid parsed time
        future_time = datetime.now() + timedelta(hours=2)
        info.set_parsed_time(future_time)
        assert info.effective_retry_time == future_time
        
        # Case 2: Fallback with calculated time
        info.use_fallback = True
        info.calculated_retry_time = datetime.now() + timedelta(hours=1)
        assert info.effective_retry_time == info.calculated_retry_time
        
        # Case 3: No parsed or calculated, use strategy
        info.parsed_retry_time = None
        info.calculated_retry_time = None
        retry_time = info.effective_retry_time
        assert retry_time > datetime.now()
    
    def test_retry_strategy_integration(self):
        """Test integration with retry strategy."""
        strategy = RetryStrategy(
            initial_delay_seconds=120,
            jitter_factor=0.0
        )
        
        info = UsageLimitRetryInfo(
            message="Limit reached",
            retry_strategy=strategy
        )
        
        # Should use strategy's delay
        info.set_parsed_time(None)
        
        retry_time = info.effective_retry_time
        expected_time = datetime.now() + timedelta(seconds=120)
        
        # Should be close (within 1 second for test timing)
        diff = abs((retry_time - expected_time).total_seconds())
        assert diff < 1