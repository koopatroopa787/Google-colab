#!/usr/bin/env python3
"""
Quick test script to verify Box2D installation

Run this to check if Box2D is properly installed.
"""

import sys

print("="*60)
print("Box2D Installation Test")
print("="*60)
print()

# Test 1: Import Box2D
print("Test 1: Importing Box2D...")
try:
    import Box2D
    print("✓ Box2D module imported successfully")
    print(f"  Version: {Box2D.__version__ if hasattr(Box2D, '__version__') else 'Unknown'}")
except ImportError as e:
    print(f"✗ Failed to import Box2D: {e}")
    print()
    print("Box2D is not installed. See INSTALL_BOX2D.md for instructions.")
    sys.exit(1)

print()

# Test 2: Import gymnasium
print("Test 2: Importing gymnasium...")
try:
    import gymnasium as gym
    print("✓ Gymnasium imported successfully")
except ImportError as e:
    print(f"✗ Failed to import gymnasium: {e}")
    print()
    print("Install with: pip install gymnasium")
    sys.exit(1)

print()

# Test 3: Create BipedalWalker environment
print("Test 3: Creating BipedalWalker-v3 environment...")
try:
    env = gym.make('BipedalWalker-v3')
    print("✓ BipedalWalker environment created successfully")

    # Test 4: Reset environment
    print()
    print("Test 4: Resetting environment...")
    obs, info = env.reset()
    print(f"✓ Environment reset successful")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")

    # Test 5: Take a step
    print()
    print("Test 5: Taking a step in the environment...")
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"✓ Step successful")
    print(f"  Reward: {reward:.2f}")
    print(f"  Terminated: {terminated}")

    env.close()

except Exception as e:
    print(f"✗ Failed to create/run BipedalWalker: {e}")
    print()
    print("This usually means Box2D is not properly installed.")
    print("See INSTALL_BOX2D.md for installation instructions.")
    sys.exit(1)

print()
print("="*60)
print("✓ All tests passed! Box2D is working correctly!")
print("="*60)
print()
print("You can now use the BipedalWalker environment:")
print("  python rl_bipedal_walker/app.py")
