from pychorus import find_and_output_chorus

input_file = "data/songs/test_blank_space.mp3"
output_file = "data/choruses/test_blank_space_chorus.wav"

find_and_output_chorus(
    input_file,
    output_file,
    15
)

print("chorus extraction complete")