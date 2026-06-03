import markovify

with (open("happyset.txt", encoding="utf-8")) as happyset:
    happy_text = happyset.read()

model = markovify.Text(happy_text, state_size=1) # State size is the number of words checked before the next word is predicted, eg. state_size = 2 would check the previous two word before preidcting

for _ in range(50):
    print(model.make_short_sentence(100))
    print("\n")