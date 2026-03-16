import babel_engine
import time

env = babel_engine.BabelEngine()

print("Engine running! Click the ground to target, and the UI to spawn.")
while True:
    env.step()
    env.pump_os_events()
    env.render()
    time.sleep(1 / 60)  # Slow it down so we can see the fall

time.sleep(1)  # Keep window open at the end
