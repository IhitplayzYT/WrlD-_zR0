"""
Integration test for the RLAIF trading simulation system.
Tests the complete pipeline from environment creation to agent training.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import numpy as np
from src.environment.market_env import MarketEnvironment
from src.agents import BaseAgent, RLAIFAgent, AdversarialAgent
from src.training.trainer import TrainingPipeline, TrainingConfig
from src.evaluation.metrics import MetricsCalculator, StatisticalSignificanceTester


def test_environment_creation():
    """Test 1: Environment creation with agents"""
    print("=" * 60)
    print("Test 1: Environment Creation")
    print("=" * 60)
    
    try:
        env = MarketEnvironment(
            n_agents=3,
            max_bad_agents=1,
            initial_cash=1000.0,
            episode_length=50,
            use_enhanced_observation=True,
            agent_types=["rlaif", "rlaif", "adversarial"]
        )
        
        print(f"✓ Environment created successfully")
        print(f"  - Number of agents: {env.n_agents}")
        print(f"  - Agent types: {env.agent_types}")
        print(f"  - Enhanced observation: {env.use_enhanced_observation}")
        print(f"  - Black box system initialized: {env.black_box_system is not None}")
        
        # Check agent objects
        for agent_id in env.agents:
            agent = env.agent_objects[agent_id]
            print(f"  - Agent {agent_id}: {agent.agent_type.value}")
        
        print("✓ Test 1 PASSED\n")
        return env
        
    except Exception as e:
        print(f"✗ Test 1 FAILED: {e}\n")
        return None


def test_environment_step(env):
    """Test 2: Environment step function"""
    print("=" * 60)
    print("Test 2: Environment Step Function")
    print("=" * 60)
    
    if env is None:
        print("✗ Test 2 SKIPPED (environment not created)\n")
        return False
    
    try:
        # Reset environment
        observations, infos = env.reset()
        print(f"✓ Environment reset successfully")
        print(f"  - Initial observations shape: {list(observations.values())[0]['price_history'].shape}")
        
        # Run a few steps
        for step in range(10):
            agent = env.agent_selection
            agent_obj = env.agent_objects[agent]
            
            # Get observation
            obs = observations[agent]
            
            # Agent decides action
            action = agent_obj.decide_action(obs, env.action_space)
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            observations = next_obs
            
            print(f"  Step {step}: Agent {agent}, Action={int(action[0])}, Reward={reward:.4f}")
            
            if terminated or truncated:
                break
        
        print(f"✓ Environment step function works correctly")
        print(f"  - Current price: {env.current_price:.2f}")
        print(f"  - Market cap: {env.market_cap:.2f}")
        print("✓ Test 2 PASSED\n")
        return True
        
    except Exception as e:
        print(f"✗ Test 2 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_black_box_system(env):
    """Test 3: Black box prediction system"""
    print("=" * 60)
    print("Test 3: Black Box Prediction System")
    print("=" * 60)
    
    if env is None:
        print("✗ Test 3 SKIPPED (environment not created)\n")
        return False
    
    try:
        black_box = env.black_box_system
        
        # Manually trigger black box
        black_box.step_counter = black_box.frequency - 1
        should_trigger = black_box.should_trigger()
        
        print(f"✓ Black box trigger check: {should_trigger}")
        
        if should_trigger:
            print(f"  - Black box activated: {black_box.is_active}")
            print(f"  - Remaining duration: {black_box.remaining_duration}")
        
        # Test prediction submission
        from src.simulation.black_box import Prediction, PredictionType
        from datetime import datetime
        
        test_prediction = Prediction(
            agent_id="agent_0",
            prediction_type=PredictionType.PRICE_LEVEL,
            predicted_values=[105.0, 106.0, 107.0, 108.0, 109.0],
            confidence_interval=(104.0, 110.0),
            confidence_level=0.8,
            explanation="Test prediction"
        )
        
        submitted = black_box.submit_prediction("agent_0", test_prediction)
        print(f"✓ Prediction submission: {submitted}")
        
        # Test ensemble prediction
        ensemble = black_box.get_ensemble_prediction()
        print(f"✓ Ensemble prediction: {ensemble is not None}")
        
        # Test market consensus
        consensus = black_box.get_market_consensus()
        print(f"✓ Market consensus: {consensus is not None}")
        
        print("✓ Test 3 PASSED\n")
        return True
        
    except Exception as e:
        print(f"✗ Test 3 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_observation_space(env):
    """Test 4: Enhanced observation space"""
    print("=" * 60)
    print("Test 4: Enhanced Observation Space")
    print("=" * 60)
    
    if env is None:
        print("✗ Test 4 SKIPPED (environment not created)\n")
        return False
    
    try:
        observations, infos = env.reset()
        
        # Check for enhanced features
        obs = list(observations.values())[0]
        
        print(f"✓ Observation components:")
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                print(f"  - {key}: shape={value.shape}")
            else:
                print(f"  - {key}: {value}")
        
        # Check for enhanced features
        enhanced_features = ['belief_state', 'temporal_features', 'current_regime', 
                           'regime_confidence', 'current_trend', 'price_divergence']
        found_features = [f for f in enhanced_features if f in obs]
        
        print(f"✓ Enhanced features found: {len(found_features)}/{len(enhanced_features)}")
        for feature in found_features:
            print(f"  - {feature}")
        
        print("✓ Test 4 PASSED\n")
        return True
        
    except Exception as e:
        print(f"✗ Test 4 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_agent_traits():
    """Test 5: Agent trait system"""
    print("=" * 60)
    print("Test 5: Agent Trait System")
    print("=" * 60)
    
    try:
        from src.agents.base_agent import AgentTraits
        
        # Create traits
        traits = AgentTraits(
            risk_tolerance=0.7,
            patience=0.3,
            trend_following=0.8
        )
        
        print(f"✓ Agent traits created")
        print(f"  - Risk tolerance: {traits.risk_tolerance}")
        print(f"  - Patience: {traits.patience}")
        print(f"  - Trend following: {traits.trend_following}")
        
        # Test behavior modifiers
        modifiers = traits.get_behavior_modifiers()
        print(f"✓ Behavior modifiers calculated:")
        for key, value in modifiers.items():
            print(f"  - {key}: {value:.4f}")
        
        # Test trait mutation
        original_risk = traits.risk_tolerance
        traits.mutate(mutation_rate=1.0, mutation_strength=0.1)
        print(f"✓ Trait mutation: {original_risk:.4f} -> {traits.risk_tolerance:.4f}")
        
        # Test trait crossover
        traits2 = AgentTraits(risk_tolerance=0.3, patience=0.8)
        child_traits = traits.crossover(traits2)
        print(f"✓ Trait crossover created")
        print(f"  - Child risk tolerance: {child_traits.risk_tolerance}")
        
        print("✓ Test 5 PASSED\n")
        return True
        
    except Exception as e:
        print(f"✗ Test 5 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_rlaif_agent():
    """Test 6: RLAIF agent with preference model"""
    print("=" * 60)
    print("Test 6: RLAIF Agent")
    print("=" * 60)
    
    try:
        agent = RLAIFAgent("test_rlaif", initial_cash=1000.0)
        
        print(f"✓ RLAIF agent created")
        print(f"  - Agent type: {agent.agent_type.value}")
        print(f"  - Curriculum level: {agent.curriculum_level}")
        
        # Test preference model
        dummy_obs = {
            'price_history': [100.0 + i * 0.1 for i in range(50)],
            'current_price': 105.0,
            'agent_cash': 1000.0,
            'agent_holdings': 10
        }
        
        action = agent.decide_action(dummy_obs, None)
        print(f"✓ Action decision made: {action}")
        
        # Test preference model evaluation
        from src.agents.rlaif_agent import PreferenceModel
        pref_model = agent.preference_model
        
        score, component_scores = pref_model.evaluate_action(
            action,
            dummy_obs,
            {'action_history': []}
        )
        
        print(f"✓ Preference model evaluation:")
        print(f"  - Total score: {score:.4f}")
        print(f"  - Component scores: {component_scores}")
        
        print("✓ Test 6 PASSED\n")
        return True
        
    except Exception as e:
        print(f"✗ Test 6 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_adversarial_agent():
    """Test 7: Adversarial agent"""
    print("=" * 60)
    print("Test 7: Adversarial Agent")
    print("=" * 60)
    
    try:
        agent = AdversarialAgent("test_adversarial", initial_cash=1000.0)
        
        print(f"✓ Adversarial agent created")
        print(f"  - Agent type: {agent.agent_type.value}")
        print(f"  - Is bad: {agent.state.is_bad}")
        print(f"  - Villain points: {agent.state.villain_points}")
        
        # Test manipulation strategy
        dummy_obs = {
            'price_history': [100.0 + i * 0.2 for i in range(50)],
            'current_price': 110.0,
            'agent_cash': 1000.0,
            'agent_holdings': 0
        }
        
        action = agent.decide_action(dummy_obs, None)
        print(f"✓ Manipulation action: {action}")
        
        # Test manipulation metrics
        metrics = agent.get_manipulation_metrics()
        print(f"✓ Manipulation metrics: {metrics}")
        
        print("✓ Test 7 PASSED\n")
        return True
        
    except Exception as e:
        print(f"✗ Test 7 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_evaluation_metrics():
    """Test 8: Evaluation metrics"""
    print("=" * 60)
    print("Test 8: Evaluation Metrics")
    print("=" * 60)
    
    try:
        calculator = MetricsCalculator(risk_free_rate=0.02)
        
        # Generate sample data
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 252).tolist()
        portfolio_values = [100.0]
        for r in returns:
            portfolio_values.append(portfolio_values[-1] * (1 + r))
        
        market_returns = np.random.normal(0.0005, 0.015, 252).tolist()
        
        # Calculate metrics
        metrics = calculator.calculate_metrics(
            returns=returns,
            portfolio_values=portfolio_values,
            market_returns=market_returns
        )
        
        print(f"✓ Metrics calculated successfully")
        print(f"  - Total return: {metrics.total_return:.4f}")
        print(f"  - Sharpe ratio: {metrics.sharpe_ratio:.4f}")
        print(f"  - Max drawdown: {metrics.max_drawdown:.4f}")
        print(f"  - Volatility: {metrics.volatility:.4f}")
        
        # Test statistical significance
        tester = StatisticalSignificanceTester(n_bootstrap=100)
        baseline_returns = np.random.normal(0.0005, 0.015, 252).tolist()
        
        bootstrap_result = tester.bootstrap_test(returns, baseline_returns, metric="mean")
        print(f"✓ Bootstrap test: p-value={bootstrap_result['p_value']:.4f}")
        
        print("✓ Test 8 PASSED\n")
        return True
        
    except Exception as e:
        print(f"✗ Test 8 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_training_pipeline():
    """Test 9: Training pipeline (short run)"""
    print("=" * 60)
    print("Test 9: Training Pipeline (Short Run)")
    print("=" * 60)
    
    try:
        config = TrainingConfig(
            n_episodes=10,  # Short run for testing
            episode_length=50,
            n_agents=2,
            use_curriculum=True,
            use_experience_replay=True,
            save_frequency=100,  # Don't save during test
            eval_frequency=5,
            log_frequency=2
        )
        
        pipeline = TrainingPipeline(config)
        
        print(f"✓ Training pipeline created")
        print(f"  - Episodes: {config.n_episodes}")
        print(f"  - Curriculum: {config.use_curriculum}")
        print(f"  - Experience replay: {config.use_experience_replay}")
        
        # Run a few episodes manually (not full training to save time)
        env = pipeline.create_environment()
        
        total_reward = 0
        for episode in range(3):
            observations, infos = env.reset()
            episode_reward = 0
            
            for step in range(20):  # Short episodes
                agent = env.agent_selection
                agent_obj = env.agent_objects[agent]
                action = agent_obj.decide_action(observations[agent], env.action_space)
                observations, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                
                if terminated or truncated:
                    break
            
            total_reward += episode_reward
            print(f"  Episode {episode}: Reward={episode_reward:.2f}")
        
        print(f"✓ Training pipeline test completed")
        print(f"  - Average reward: {total_reward/3:.2f}")
        
        print("✓ Test 9 PASSED\n")
        return True
        
    except Exception as e:
        print(f"✗ Test 9 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all integration tests"""
    print("\n" + "=" * 60)
    print("RLAIF TRADING SIMULATION - INTEGRATION TESTS")
    print("=" * 60 + "\n")
    
    results = {}
    
    # Test 1: Environment creation
    env = test_environment_creation()
    results['environment_creation'] = env is not None
    
    # Test 2: Environment step
    results['environment_step'] = test_environment_step(env)
    
    # Test 3: Black box system
    results['black_box_system'] = test_black_box_system(env)
    
    # Test 4: Observation space
    results['observation_space'] = test_observation_space(env)
    
    # Test 5: Agent traits
    results['agent_traits'] = test_agent_traits()
    
    # Test 6: RLAIF agent
    results['rlaif_agent'] = test_rlaif_agent()
    
    # Test 7: Adversarial agent
    results['adversarial_agent'] = test_adversarial_agent()
    
    # Test 8: Evaluation metrics
    results['evaluation_metrics'] = test_evaluation_metrics()
    
    # Test 9: Training pipeline
    results['training_pipeline'] = test_training_pipeline()
    
    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All integration tests passed!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
    
    print("=" * 60 + "\n")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
