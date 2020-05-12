"""Attempt at the Semantics for Formality Style
Handcrafted list from 
E&M/test/formal
E&M/test/informal.rule_based
E&M/test/informal.ref0
E&M/test/informal.ref1
E&M/test/informal.ref2
E&M/test/informal.ref4
"""

index2feature = {
    0: "education_level", 
    1: "emotions", 
    2: "audience_relationship"
}

feature2index = {val: key for key, val in index2feature.items()}

EMOTIONAL_WORDS = {"suck", "like", "love", "hate", "dumbest", "dumb", "nice", 
"cute", "dare", "stupid", "awesome", "funny", "sad", "luv", "freaking", "miss"}

EMOTICONS = {":))", ":)", ":(", ":/", ":D", ":P", "<3"}

SLANG_WORDS = {"cuz", "bcuz", "juz", "n", "dat", "luv", "dis", "r", "u", "coz", 
"aint", "ppl", "y", "ur", "wat", "wut", "cud", "lol", "etc", "lul", "yo", "yeah", 
"hehe", "some1", "4", "yap", "ya", "yeh", "omg", "bcoz", "b4", "cmon", "freaking",
"eww", "ew", "freakin", "nah"}

SWEAR_WORDS = {"damn", "arse", "ass", "asshole", "bastard", "bitch", "bollocks",
"bugger", "cunt", "damn", "effing", "frigger", "fuck", "goddamn", "godsdamn", 
"hell", "horseshit", "motherfucker", "nigga", "nigger", "prick", "shit", "shitass"
"slut", "twat", "yikes", "grrrr"}

CONTRACTIONS = {"isn't", "aren't", "don't", "doesn't", "couldn't", "can't", 
                "hasn't", "haven't", "i'm", "it'd", "it's", "they're", "he's", 
                "she's", "there's"}

COMMON_PUNCTUATIONS = {".", ";", ","}

PROPER_NOUNS = {"steve", "martin", "beyonce", "melania", "youtube.com", "rihanna",
"hazzard", "simpsons", "oprah", "genesis", "emo", "aaliyah", "aries", "bjork", 
"chaney", "brokeback", "anakin", "amadala", "buffalo", "ny", "paula", "smallville", 
"RIAA", "beastie", "dylan", "gonzales", "tom", "jerry", "techno", "milli", "vanilli",
"stephen", "n'sync", "ebay", "elm", "simon", "arkansas", "mickey", "nicole", "kidman",
"jason", "momoa", "baywatch", "chris", "stacie", "1001", "disney", "40yr", "josh", 
"groban", "kathy", "bates", "aladdin", "jessica", "paris", "hilton", "vince", "vaughn",
"donnie", "darko", "choirboy", "hollywood", "mariah", "careys", "chely", "wright", 
"drat", "warcraft", "mccartney", "lindsay", "lohan", "zac", "efron", "taboobuster", 
"osama", "beavis", "itunes.com", "limeware.com", "cheeh", "chongs", "floyd", "bambi", 
"brad", "pitt", "pippin", "johnny", "depp", "chrons", "tom", "cruise", "jessica", 
"tonto", "alan", "moore", "orlando", "bloom", "chad", "michael", "murray", "joan", 
"joker", "batman", "carmen", "electra", "jodie", "foster", "chuck", "norris", "jess", 
"diane", "lane", "adam", "sandler", "rob", "thomas", "gilmore", "doyle", "brunsen", 
"boyz", "II", "will", "farrell", "colin", "powell", "yoko", "tyra", "banks", "charlie's",
"hoopz", "flav", "kevin", "ac/dc", "james", "angelina", "jolie", "sarah", "christian",
"tinkerbell", "ram", "geeta", "st.", "ives", "peggy", "sue", "screamo", "google", 
"blockbuster", "lucius", "malfoy", "playstation", "(PSP)", "jennifer", "hudson", 
"chuby", "checkers", "michael", "jackson", "scientology", "bubblicious", "kryptonite", 
"rie-rie", "mandisa", "shakespear", "romeo", "juliet", "george", "clooney", "lauren", 
"katharine", "katie", "mr", "kapony", "mary", "blige", "karishma", "kapoor", "scientologist", 
"holmes", "sam", "bollywood", "beatles", "naruto", "bugle's", "jennifer", "hewitt", 
"abbie", "mitchell", "clara", "broadway", "kazza", "limewire", "eztrackz", "aspen", 
"colorado", "john", "wayne", "bruce", "dern", "coyboys", "missouri", "st.", "louis", 
"eminem", "mystic", "pizza", "ps.", "raveena", "tandon's", "taylor", "hicks", 
"incredibles", "dookie", "reese", "witherspoon", "goo", "aishwarya", "rai", "'lp'", 
"'linkin", "park'", "tlc", "baldwin", "bewitched", "elf", "askjeeves", "scorpio", 
"jeannie", "u2", "bon", "jovi", "coldplay", "rudy", "theo", "manchurian", "mona", "lisa", 
"omarion", "chrison", "bow", "wow", "superman", "norton", "antivirus", "billy", "joel's", 
"janis", "joplin's", "snoopy", "rachel", "fuller", "julia", "roberts", "keith",
"randy", "harry", "potter", "texas", "alba", "mickey", "rourke", "bruce", "willis", 
"screamo", "panget", "natasha", "bedingfield", "james", "blunt", "elvis", "christopher", 
"reeves", "ne-yo", "radiohead", "Filipinos", "RBD", "jim", "carrey", "peter", "susan", 
"lucy", "edumnd", "greenday", "costner", "schindlers", "archie", "betty", "taylore", 
"mandisa", "lisa", "robin", "aniston", "kellie", "morgan", "bel-air", "ABBA", "watson", 
"R&B"}