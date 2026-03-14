import json, random

templates = [
    ("The product stopped working after {n} days.", "We're sorry to hear that. Please contact support@company.com and we'll replace it within 48 hours."),
    ("Delivery took {n} weeks, too slow.", "We apologize for the delay. Your next order gets free express shipping automatically."),
    ("Best purchase I've made in {n} years!", "Thank you so much! We're thrilled it's working well for you."),
    ("The quality is poor for the price of ${n}.", "We appreciate the honest feedback. We've forwarded this to our quality team."),
    ("Instructions were confusing, took {n} tries.", "Sorry for the frustration! We're updating the manual. Here's a video guide: link."),
    ("Arrived damaged after {n} days in transit.", "That's unacceptable. We'll ship a replacement today at no cost."),
    ("Works perfectly after {n} months of use.", "Wonderful to hear! Built to last is our goal."),
    ("Customer service took {n} days to respond.", "We're improving response times. You're right to be frustrated — thank you for the patience."),
]

data = []
for i in range(500):
    template = random.choice(templates)
    n = random.randint(1, 30)
    review = template[0].format(n=n)
    response = template[1]
    data.append({
        "instruction": "You are a professional customer support agent. Respond to this review:",
        "input": review,
        "output": response
    })

with open("reviews_dataset.json", "w") as f:
    json.dump(data, f, indent=2)

print(f"Generated {len(data)} samples")