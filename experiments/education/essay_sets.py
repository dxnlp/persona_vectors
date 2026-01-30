"""
ASAP-SAS Essay Set Configurations with prompts, contexts, and rubrics.

Each set contains:
- prompt: The question/task for the student
- context: Reading passage or experiment description (if applicable)
- rubric: Scoring criteria for the judge
- score_range: (min, max) scores
- topic: Brief description
"""

ESSAY_SETS = {
    1: {
        "topic": "Acid Rain - Science Experiment",
        "score_range": (0, 3),
        "context": """A group of students wrote the following procedure for their investigation.

Procedure:
1. Determine the mass of four different samples.
2. Pour vinegar in each of four separate, but identical, containers.
3. Place a sample of one material into one container and label. Repeat with remaining samples, placing a single sample into a single container.
4. After 24 hours, remove the samples from the containers and rinse each sample with distilled water.
5. Allow the samples to sit and dry for 30 minutes.
6. Determine the mass of each sample.

The students' data are recorded in the table below:
|iteite |ite Massite (ite g) Before | ite Mass (g) After |
|ite --|ite --|ite --|
|iteIteIte Limestone ite | ite ite ite ite ite ite 10.0 ite | ite ite ite ite ite ite ite ite ite ite ite ite ite ite 9.7 ite |
| ite Marble ite | ite ite ite ite ite 10.0 ite | ite ite ite ite ite ite ite ite ite ite ite ite ite 9.8 ite |
| ite Wood ite | ite ite ite ite ite ite 10.0 ite | ite ite ite ite ite ite ite ite ite ite ite ite ite 10.0 ite |
|Ite Plastic ite | ite ite ite ite ite 10.0 ite | ite ite ite ite ite ite ite ite ite ite ite ite ite 10.0 ite |""",
        "prompt": "After reading the group's procedure, describe what additional information you would need in order to replicate the experiment. Make sure to include at least three pieces of information.",
        "rubric": """Score 3: The response describes three or more additional pieces of information needed to replicate the experiment (e.g., amount of vinegar, type of vinegar, size of samples, drying method, container type).

Score 2: The response describes two additional pieces of information needed to replicate the experiment.

Score 1: The response describes one additional piece of information needed to replicate the experiment.

Score 0: The response describes little or no accurate or relevant information about what is needed to replicate the experiment.""",
    },

    2: {
        "topic": "Polymer Investigation - Science Experiment",
        "score_range": (0, 3),
        "context": """A student performed the following investigation to test four different polymer plastics for stretchability.

Procedure:
1. Take a sample of one type of plastic, and measure its length.
2. Tape the top edge of the plastic sample to a table so that it is hanging freely down the side of the table.
3. Attach a clamp to the bottom edge of the plastic sample.
4. Add weights to the clamp and allow them to hang for five minutes.
5. Remove the weights and clamp, and measure the length of the plastic types.
6. Repeat the procedure exactly for the remaining three plastic samples.
7. Perform a second trial (T2) exactly like the first trial (T1).

Data Table:
| Plastic | Starting Length (cm) | Trial 1 Final Length (cm) | Trial 2 Final Length (cm) |
|---------|---------------------|---------------------------|---------------------------|
| A       | 10                  | 11                        | 11                        |
| B       | 10                  | 15                        | 14                        |
| C       | 10                  | 11                        | 12                        |
| D       | 10                  | 14                        | 15                        |""",
        "prompt": "Draw a conclusion based on the student's data. Describe two ways the student could have improved the experimental design and/or validity of the results.",
        "rubric": """Score 3: The response draws a valid conclusion supported by the data AND describes two ways to improve the experimental design/validity.

Score 2: The response draws a valid conclusion AND describes one improvement, OR describes two improvements but fails to draw a valid conclusion.

Score 1: The response draws a valid conclusion but fails to describe improvements, OR describes one improvement but fails to draw a valid conclusion.

Score 0: The response provides little or no correct information from the investigation.""",
    },

    3: {
        "topic": "Koala/Panda - Reading Comprehension",
        "score_range": (0, 2),
        "context": """THE WASHINGTON POST FRIDAY, APRIL 18, 2008
One Man's Pet, Another's Invasive Species
BY JOEL ACHENBACH

BUSHNELL, Fla—RobRoy MacInnes is the man to see if you want to buy a crocodile. Or a scorpion, a rattlesnake, a boa constrictor. Got a hankering for a cobra? Just pony up $600 and you can have one of the more lethal species.

MacInnes is co-owner of Glades Herp Farms, an empire of claws, spines, scales, fangs and darting tongues. The reptile trade, he's happy to report, is booming. The pet industry estimates that about 4.8 million households now contain at least one pet reptile, a number that has nearly doubled in a decade.

But biologists see the trade in nonnative creatures as a factor in the rising number of invasive species, such as the Burmese python, which is breeding up a storm in the Everglades, and the Nile monitor lizard, a toothy carnivore that can reach seven feet in length.

MacInnes contends that the government overestimates the threat posed by invasive reptiles. Even the term "invasive species" is unfair, he said. "They're 'introduced.' I think that 'invasive' is passing judgment."

Of the pythons, he said: "To me, it's a wonderful introduction. I think it's the best thing to happen to the Everglades in the last 200 years."

Biologists, however, say that invasive species, unchecked by natural predators, are major threats to biodiversity. Life on Earth has always moved around, but never so fast. Organisms evolve in niche environments. What happens when the natural barriers are removed? When anything can go anywhere? Complications ensue.

Skip Snow, a wildlife biologist for Everglades National Park, has helped drag hundreds of Burmese pythons out of the weeds. He calls MacInnes's argument "ridiculous." The snakes, he says, are imperiling five endangered species in the Florida Keys.

What is happening in Florida illustrates a broader fact about life on Earth: We live in an age that favors generalists rather than specialists. A generalist is a raccoon, a python, a cockroach, a white-tailed deer. The ultimate generalist is, arguably, a human being, who with the assistance of technology can live anywhere from Florida to Antarctica to outer space.

A specialist is China's panda, which eats almost nothing but bamboo, or Australia's koala bear, which eats eucalyptus leaves almost exclusively.

MacInnes is not without an environmental conscience. "We're degrading the Earth at an alarming rate," he said. He added: "What favors generalists is change. What favors specialists is stability. Right now, mankind has chosen to make Earth a rapidly changing place."

Down in the Everglades, Skip Snow would agree with that part of MacInnes's philosophy. We are all part of a vast experiment in the blending of organisms from around the world, he said. "The thing about the experiment is, it's not planned, and there's no one in control," Snow added. "It's an experiment run amok." """,
        "prompt": "Explain how pandas in China are similar to koalas in Australia and how they both are different from pythons. Support your response with information from the article.",
        "rubric": """Score 2: The response demonstrates strong understanding with specific relevant information from the text. Shows that pandas and koalas are both specialists (eat specific foods - bamboo and eucalyptus) while pythons are generalists. May include insights about how specialists need stability while generalists thrive on change.

Score 1: The response demonstrates some understanding with some relevant information. May identify that pandas and koalas are similar or that pythons are different, but lacks depth or specific text support.

Score 0: The response demonstrates limited or no understanding. May be vague, irrelevant, or contain inaccurate information.""",
    },

    4: {
        "topic": "Invasive Species - Reading Critical Analysis",
        "score_range": (0, 2),
        "context": """THE WASHINGTON POST FRIDAY, APRIL 18, 2008
One Man's Pet, Another's Invasive Species
BY JOEL ACHENBACH

BUSHNELL, Fla—RobRoy MacInnes is the man to see if you want to buy a crocodile. Or a scorpion, a rattlesnake, a boa constrictor. Got a hankering for a cobra? Just pony up $600 and you can have one of the more lethal species.

MacInnes is co-owner of Glades Herp Farms, an empire of claws, spines, scales, fangs and darting tongues. The reptile trade, he's happy to report, is booming. The pet industry estimates that about 4.8 million households now contain at least one pet reptile, a number that has nearly doubled in a decade.

But biologists see the trade in nonnative creatures as a factor in the rising number of invasive species, such as the Burmese python, which is breeding up a storm in the Everglades, and the Nile monitor lizard, a toothy carnivore that can reach seven feet in length.

MacInnes contends that the government overestimates the threat posed by invasive reptiles. Even the term "invasive species" is unfair, he said. "They're 'introduced.' I think that 'invasive' is passing judgment."

Of the pythons, he said: "To me, it's a wonderful introduction. I think it's the best thing to happen to the Everglades in the last 200 years."

Biologists, however, say that invasive species, unchecked by natural predators, are major threats to biodiversity. Life on Earth has always moved around, but never so fast. Organisms evolve in niche environments. What happens when the natural barriers are removed? When anything can go anywhere? Complications ensue.

Skip Snow, a wildlife biologist for Everglades National Park, has helped drag hundreds of Burmese pythons out of the weeds. He calls MacInnes's argument "ridiculous." The snakes, he says, are imperiling five endangered species in the Florida Keys, including the Key Largo wood rat, one specimen of which, tagged with a radio transmitter, was tracked all the way to the belly of a python.

No one knows how the snakes went native, but there's speculation that Hurricane Andrew played a factor in a wholesale python jailbreak in 1992. Even more dramatic was what happened in the Everglades in 2005: A python swallowed an alligator and exploded. The photograph ran around the world.

"Ecophobia is playing a role," said Jamie K. Reaser, a science and policy adviser. "Mammals are warm and fuzzy. Birds tend to have quite a following. But animals such as lizards and snakes tend, at least in this culture, to be less well respected."

Down in the Everglades, Skip Snow said we are all part of a vast experiment in the blending of organisms from around the world. "The thing about the experiment is, it's not planned, and there's no one in control," Snow added. "It's an experiment run amok." """,
        "prompt": "Explain the significance of the word 'invasive' to the rest of the article. Support your response with information from the article.",
        "rubric": """Score 2: The response demonstrates thoughtful examination of the text. Shows understanding that "invasive" is a contested term - MacInnes prefers "introduced" while biologists see invasive species as threats. May evaluate the author's presentation of both perspectives.

Score 1: The response demonstrates sufficient but limited examination. May identify that invasive relates to non-native species or that there's debate about the term, but lacks depth.

Score 0: The response demonstrates cursory examination with little or no ability to evaluate the significance of the term. May be superficial, vague, or irrelevant.""",
    },

    5: {
        "topic": "Protein Synthesis - Biology",
        "score_range": (0, 3),
        "context": """Protein synthesis is the process by which cells build proteins. The process involves two main stages: transcription (in the nucleus) and translation (in the cytoplasm). During this process, the genetic information in DNA is used to produce proteins that carry out various functions in the cell.""",
        "prompt": "Starting with mRNA leaving the nucleus, list and describe four major steps involved in protein synthesis.",
        "rubric": """Score 3: The response describes four key elements of protein synthesis, such as:
- mRNA exits nucleus via nuclear pore
- mRNA travels to ribosome
- mRNA codons are read in triplets
- tRNA brings complementary anticodons
- Amino acids are bonded together
- Process continues until STOP codon

Score 2: The response describes three key elements.

Score 1: The response describes one or two key elements.

Score 0: The response contains no accurate key elements or is off-topic.""",
    },

    6: {
        "topic": "Cell Membrane - Biology",
        "score_range": (0, 3),
        "context": """The cell membrane (plasma membrane) is a biological membrane that separates the interior of all cells from the outside environment. The cell membrane is selectively permeable to ions and organic molecules and controls the movement of substances in and out of cells.""",
        "prompt": "List and describe three processes used by cells to control the movement of substances across the cell membrane.",
        "rubric": """Score 3: The response describes three key processes, such as:
- Passive transport/diffusion (high to low concentration)
- Osmosis (diffusion of water)
- Facilitated diffusion (using membrane proteins)
- Active transport (using energy, low to high concentration)
- Endocytosis (engulfing materials)
- Exocytosis (expelling materials)
- Protein channels

Score 2: The response describes two key processes.

Score 1: The response describes one key process.

Score 0: The response contains no accurate processes or is off-topic.""",
    },

    7: {
        "topic": "Trait of Rose - Literature",
        "score_range": (0, 2),
        "context": """Crossing Over

Rose's head jerked up from her chest. "Oh no," she groaned, rubbing the back of her neck and blinking at the bright light in the kitchen. For a split second she was confused. Then she remembered: her essay for the state competition. She'd been struggling to think of a topic. Her brain must have surrendered to exhaustion.

The day, like most of her days, had been too long, too demanding. From school she'd gone straight to the restaurant to work a four-hour shift, then straight home to help Aunt Kolab prepare a quick supper. After that it was time to do homework.

When would she squeeze in writing a flawless three-thousand-word essay? "I'm insane," she said grimly as she gathered books and papers.

Her younger sister walked in rubbing her eyes.
"Anna," Rose said. "What's wrong? You feel okay?"
"I'm fine," her sister said. "I just had another bad dream."
"I fell asleep working on my essay," Rose said.

Anna poured two glasses of orange juice and handed one to Rose. "Mama's not home yet, is she." It wasn't a question. "I hate how late she has to work." Her voice sank to a fierce whisper. "I'm so lonesome for Papa. It seems like he's been gone for years."

"It's only been four months," Rose said as gently as she could. "He had to go. The job in Los Angeles paid three times what he was making here."

Anna glared at Rose. "Money isn't everything."

"Only if you already have everything," Rose said. She tried a laugh that sounded fake even to her. "We have our part to do to help Paul finish college. Then he'll get a good job, Anna, and he'll pay for you and me to go to college."

Anna rolled her eyes and shoved her chair away from the table. "You sound just like Mama." She stood and stalked out of the kitchen.

In the dark Rose clasped then cupped her hands. Paul's fate lies partly in these, she thought. She felt too young for so much responsibility. Then she shivered, imagining how her brother must feel. Only three years older, he held the fate of two people—both his sisters—in his hands.

The next morning, Rose stopped at her mother's room. She was sleeping. So it was Aunt Kolab making the muted noises coming from the kitchen.

"Good morning, Rose," her aunt said.

Rose felt an urgent need to relate her dream, to expose it so it would loosen its grip on her. After she'd finished, her aunt said, with a puzzled look, "Do you feel so weighed down by what you're doing to help this family?"

Rose didn't answer. If she told the truth, she would hurt her aunt.

"In Cambodia, our first country, what we're all doing would be quite normal," her aunt said. "But now I realize that you're seeing the situation through other eyes—as you should, I suppose, because you grew up here…. This must be difficult for you. Yes?"

Rose nodded.

"Hmm. Maybe we can find a way to do things differently. A way better for you." Her aunt's face lit up. "Maybe I can sew for ladies. Or I could make special treats from our country and sell them."

Rose kept nodding. Maybe her life would get easier. Maybe it wouldn't. But her aunt's offer had somehow made her feel lighter. Suddenly, it occurred to her: here was the topic for her essay. Cambodian tradition and sense of family, she realized, could survive an ocean crossing.""",
        "prompt": "Identify ONE trait that can describe Rose based on her conversations with Anna or Aunt Kolab. Include ONE detail from the story that supports your answer.",
        "rubric": """Score 2: The response identifies a valid character trait of Rose (e.g., responsible, caring, mature, hardworking, supportive, understanding) AND includes a specific detail from the story that supports it.

Score 1: The response identifies a valid character trait of Rose but lacks supporting detail, or the detail is vague/general.

Score 0: The response does not identify a valid character trait of Rose, or the trait is unsupported and inaccurate.""",
    },

    8: {
        "topic": "Mr. Leonard - Literature",
        "score_range": (0, 2),
        "context": """Gifts

I met Mr. Leonard when I started middle school. He was a hall monitor whose job it was to keep students moving along from one classroom to the next. "Move along, people, move along!" he'd advise the shuffling crowd.

I distinguished myself from the masses by being one of a select few in the remedial reading program. Twice a week, I left English class early for the learning center in the basement, where I worked with a tutor. On my first trip, Mr. Leonard confronted me in the stairwell.

"Hey, my friend, where do you think you're going?" he asked, arms folded across his chest.
"Learning center," I muttered, showing him my hall pass.
"Why?" he asked from behind a hard stare.
"Why?" I answered automatically, "I can't read."
His gaze softened. "Fair enough. On your way, then. Work hard."

Then one day he surprised me by asking what I did after school.
"Nothing," I answered. "Just some homework."
"Meet me in the gym. 2:30."

When I arrived, the gym was crowded with kids warming up for intramurals. Mr. Leonard was seated in a corner, watching. He waved me over, then pointed at the kids chasing basketballs. "None of this appeals to you?" he asked.

I shook my head. When you're the last guy chosen for teams in gym class, you don't seek out more of that treatment after school.

"Follow me," he directed. We left the building and went to the track. Spread along the inside lane were hurdles. Mr. Leonard pointed at the closest one.
"Know what that thing is called?" he asked.
"A hurdle," I answered.
"Know what to do with it?" he questioned.
"You jump it," I replied.
"Well?" he responded. "On your way then."

And so it began. Monday through Friday, rain or shine, I was out on the track with Mr. Leonard shouting from the side. "Open your stride!" "Pump your arms!" "Lean,... NOW!" I improved steadily until one day I found myself standing before the high school track coach.

"How'd you get so fast, son?" he asked.
"Well, I've been training," I replied. "Someone's helping me."
"Mr. Leonard Grabowski?"

The coach handed me a URL for a track and field website. "Visit this site. Do a search for 'Grabowski.'"

The next day, I showed Mr. Leonard the printout I'd downloaded. "Why didn't you tell me about this?"

He smiled sadly at the image on the page. "I looked good back then, didn't I?" he chuckled.

I pointed to the photograph. "You were a college freshman who won the 400 meter hurdles at the nationals. You broke records."

"I remember," he said solemnly. "Best race of my life."

"Well, what happened after that?" I pressed.

Mr. Leonard's voice cracked as he spoke. "I was a good athlete," he said softly, "but not a good student. We had no learning centers in our school. I relied on friends to help me get by, but even then the work was always too hard." His voice trailed off.

"But you went to college," I said.

"The college scouts told me that my grades didn't matter, that I'd have tutors to help me, but college work is a whole lot harder than high school work. I lost my scholarship and flunked out. No other school wanted a runner who couldn't read."

The emotions in Mr. Leonard's words were all too familiar to me. I knew them well—feelings of embarrassment when I was called upon to read aloud. This man had given his time to help me excel at something. Suddenly I realized what I could do for him.

"C'mon, Mr. Leonard," I said, walking back toward school. "It's time to start your training." """,
        "prompt": "During the story, the reader gets background information about Mr. Leonard. Explain the effect that background information has on Paul. Support your response with details from the story.",
        "rubric": """Score 2 (Proficient): The response explains how learning about Mr. Leonard's past (being a champion athlete who couldn't read and lost his scholarship) affects Paul. Shows that Paul feels connected to Mr. Leonard's struggle and decides to help him learn to read in return. Includes specific story details.

Score 1 (Partially Proficient): The response addresses the effect but may be too general, too simplistic, or lacks sufficient text support.

Score 0 (Not Proficient): The response does not explain the effect, or contains inaccurate/incomplete information.""",
    },

    9: {
        "topic": "Organization of Article - Reading Informational",
        "score_range": (0, 2),
        "context": """Orbiting Junk

Grab your telescope! Look up in the sky! It's a comet! It's a meteor! It's a... tool bag?

Such an observation isn't as strange as it seems. Orbital pathways around our planet that were once clear are now cluttered with the remains of numerous space exploration and satellite missions. This "space junk" is currently of great concern to government space agencies around the globe.

What Is Space Junk?
In 1957, the Soviet Union launched the first artificial satellite. The United States followed suit, and thus began the human race's great space invasion.

Over the past 52 years, a variety of spacecraft, including space capsules, telescopes, and satellites, have been sent beyond Earth's atmosphere. They explore the vast reaches of our solar system, monitor atmospheric conditions, and make global wireless communication possible. The rockets that are used to power these spacecraft typically fall back to Earth and disintegrate. The objects themselves, however, are positioned hundreds of miles above Earth. In this airless environment, some of them continue to circle the planet indefinitely. While this is ideal for a fully functioning object, what happens when a satellite "dies" or malfunctions? The disabled object becomes a piece of high-tech junk, circling the globe in uncontrolled orbit.

Crash Course
With no one at the controls, dead satellites run the risk of colliding with each other. That's exactly what happened in February 2009. Two communications satellites, one American and one Russian, both traveling at more than 20,000 miles per hour, crashed into each other 491 miles above the Earth. The impact created hundreds of pieces of debris.

It's not only spectacular crashes that create debris. Any objects released into space become free-orbiting satellites. In 2008, a tool bag escaped from the grip of an astronaut doing repairs on the International Space Station.

Little Bits, But a Big Deal
So who cares about a lost tool bag or tiny bits of space trash?

Actually, many people do. Those bits of space debris present a very serious problem. Tiny fragments traveling at a speed of five miles per second can inflict serious damage on the most carefully designed spacecraft.

Scientists are hard-pressed for an easy solution. Both NASA and the European Space Agency maintain catalogues of known objects. The lost tool bag, for example, is listed as Satellite 33442. But while military radar can identify objects the size of a baseball, anything smaller goes undetected.

Yet the problem is certain to persist. The amount of space trash is actually increasing because commercial space travel is on the rise and more nations have undertaken space exploration. Space agencies hope that the corporations and nations involved can work together to come up with a viable solution to space pollution.""",
        "prompt": "How does the author organize the article? Support your response with details from the article.",
        "rubric": """Score 2 (Proficient): The response identifies the organizational structure (e.g., problem-solution, cause-effect, or sequential with headings) and supports with specific details from the article (mentions the headings like "What Is Space Junk?", "Crash Course", "Little Bits, But a Big Deal" or the progression from defining the problem to explaining causes to discussing solutions).

Score 1 (Partially Proficient): The response addresses organization but may be too general or lacks sufficient text support.

Score 0 (Not Proficient): The response does not address organization, or contains inaccurate/incomplete information.""",
    },

    10: {
        "topic": "Doghouse - Science Application",
        "score_range": (0, 2),
        "context": """Brandi and Jerry did the following controlled experiment to find out how the color of an object affects its temperature.

Question: What is the effect of different lid colors on the air temperature inside a glass jar exposed to a lamp?

Hypothesis: The darker the lid color, the greater the increase in air temperature in the glass jar, because darker colors absorb more energy.

Materials: glass jar, lamp, four colored lids (black, dark gray, light gray, and white), thermometer, meterstick, stopwatch

Procedure:
1. Put the black lid with the attached thermometer on the glass jar.
2. Make sure the starting temperature inside the jar is 24°C.
3. Place lamp 5 centimeters away from the lid and turn on the lamp.
4. After 10 minutes measure the air temperature inside the glass jar and record as Trial 1.
5. Turn off lamp and wait until the air in the jar returns to the starting temperature.
6. Repeat steps 2 through 5 for Trials 2 and 3.
7. Repeat steps 1 through 6 for the dark gray, light gray, and white lids.
8. Calculate and record the average air temperature for each lid color.

Data Table:
| Lid Color   | Trial 1 (°C) | Trial 2 (°C) | Trial 3 (°C) | Average (°C) |
|-------------|--------------|--------------|--------------|--------------|
| Black       | 35           | 36           | 35           | 35.3         |
| Dark Gray   | 32           | 31           | 32           | 31.7         |
| Light Gray  | 28           | 27           | 28           | 27.7         |
| White       | 25           | 26           | 25           | 25.3         |""",
        "prompt": """Brandi and Jerry were designing a doghouse. Use the results from the experiment to describe the best paint color for the doghouse.

In your description, be sure to:
- Choose a paint color
- Describe how that color might affect the inside of the doghouse
- Use results from the experiment to support your description""",
        "rubric": """Score 2: The response chooses a color (white or light color) AND describes how it affects the doghouse (keeps it cooler) AND uses experimental results to support (white had lowest temperature at 25.3°C).

Score 1: The response chooses a color and either describes the effect OR uses experimental results, but not both.

Score 0: The response does not choose an appropriate color, or fails to connect the choice to the experiment in any meaningful way.""",
    },
}


def get_essay_set(set_id: int) -> dict:
    """Get essay set configuration by ID."""
    return ESSAY_SETS.get(set_id, None)


def get_all_set_ids() -> list:
    """Get all available essay set IDs."""
    return list(ESSAY_SETS.keys())


def normalize_score(score: int, set_id: int) -> float:
    """Normalize a score to 0-1 range based on the set's score range."""
    min_score, max_score = ESSAY_SETS[set_id]["score_range"]
    return (score - min_score) / (max_score - min_score)


def denormalize_score(normalized: float, set_id: int) -> int:
    """Convert normalized score back to original range."""
    min_score, max_score = ESSAY_SETS[set_id]["score_range"]
    return round(normalized * (max_score - min_score) + min_score)
