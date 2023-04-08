from pypersonality.personality import PyPersonality


p = PyPersonality()
results = p.get_personality(
    "Oh, thanks for inviting me. I appreciate it. I'm not sure if I'll be able to make it, though. I have some other things I need to take care of this weekend . I'm sure it will be a great time, but I'm just not feeling up for a big social gathering right now. Maybe we can plan something else another time?"
)

print(results)
