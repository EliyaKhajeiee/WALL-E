#!/usr/bin/env python3
# test_button.py - Find your HAT button GPIO pin
"""
This script tests common GPIO pins to find which one your HAT button is on.
Press the button when prompted to identify the correct pin.
"""

from gpiozero import Button
import time

# Test common HAT button pins
test_pins = [17, 22, 23, 24, 27, 4, 5, 6, 12, 13, 16, 18, 25, 26]

print("=" * 50)
print("HAT Button Pin Detector")
print("=" * 50)
print("\nTesting GPIO pins for button...")
print("Press the button on your HAT when prompted!\n")

found = False

for pin in test_pins:
    try:
        button = Button(pin, pull_up=True, bounce_time=0.1)
        print(f"Testing GPIO {pin}... Press button now (5 sec)...")

        if button.wait_for_press(timeout=5):
            print(f"\n✓ FOUND IT! Your button is on GPIO {pin}")
            print(f"\nUpdate BUTTON_PIN = {pin} in walle_voice.py")
            found = True
            break
        else:
            print(f"  ✗ No button detected on GPIO {pin}")

    except Exception as e:
        print(f"  ✗ GPIO {pin} failed: {e}")

    time.sleep(0.5)

if not found:
    print("\n❌ Button not found on common pins!")
    print("Your HAT might use a different pin or require different configuration.")
    print("Check your HAT documentation for the button GPIO pin number.")

print("\nTest complete!")
