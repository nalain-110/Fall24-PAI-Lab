"""
Personality mapping from facial feature ratios to MBTI types.

DISCLAIMER: This is for academic/demo purposes only.
Face-to-personality mapping is NOT scientifically validated.
It is used here as a creative exercise for the assignment.

MBTI Dimensions:
  I/E  - Introvert / Extrovert
  N/S  - Intuitive / Sensing
  F/T  - Feeling / Thinking
  P/J  - Perceiving / Judging
"""

MBTI_PROFILES = {
    "INTJ": {
        "name": "The Architect",
        "description": "Strategic, independent, and determined. Natural leaders who prefer logic over emotion.",
        "traits": ["Strategic thinker", "Independent", "Decisive", "High standards"],
        "strengths": "Planning, problem-solving, long-term vision",
        "careers": "Engineer, Scientist, Strategist",
    },
    "INTP": {
        "name": "The Logician",
        "description": "Innovative, curious, and analytical. Love theoretical problems and abstract ideas.",
        "traits": ["Analytical", "Curious", "Objective", "Reserved"],
        "strengths": "Analysis, innovation, pattern recognition",
        "careers": "Mathematician, Programmer, Philosopher",
    },
    "ENTJ": {
        "name": "The Commander",
        "description": "Bold, imaginative and strong-willed leaders who find a way.",
        "traits": ["Confident", "Strategic", "Charismatic", "Driven"],
        "strengths": "Leadership, organization, efficiency",
        "careers": "Executive, Lawyer, Manager",
    },
    "ENTP": {
        "name": "The Debater",
        "description": "Smart, curious thinkers who love intellectual challenges.",
        "traits": ["Quick-witted", "Energetic", "Outspoken", "Innovative"],
        "strengths": "Brainstorming, debate, entrepreneurship",
        "careers": "Entrepreneur, Consultant, Inventor",
    },
    "INFJ": {
        "name": "The Advocate",
        "description": "Quiet and mystical, yet inspiring — driven by deep values.",
        "traits": ["Insightful", "Principled", "Empathetic", "Private"],
        "strengths": "Empathy, vision, helping others",
        "careers": "Counselor, Writer, Teacher",
    },
    "INFP": {
        "name": "The Mediator",
        "description": "Poetic, kind and altruistic. Always looking for the good in people.",
        "traits": ["Idealistic", "Empathetic", "Creative", "Flexible"],
        "strengths": "Creativity, empathy, open-mindedness",
        "careers": "Writer, Artist, Therapist",
    },
    "ENFJ": {
        "name": "The Protagonist",
        "description": "Charismatic and inspiring leaders who mesmerize their listeners.",
        "traits": ["Charismatic", "Empathetic", "Organized", "Reliable"],
        "strengths": "Communication, leadership, inspiration",
        "careers": "Teacher, Coach, HR Manager",
    },
    "ENFP": {
        "name": "The Campaigner",
        "description": "Enthusiastic, creative and sociable free spirits who find reason to smile.",
        "traits": ["Enthusiastic", "Creative", "Sociable", "Optimistic"],
        "strengths": "Communication, creativity, enthusiasm",
        "careers": "Journalist, Actor, Designer",
    },
    "ISTJ": {
        "name": "The Logistician",
        "description": "Practical and fact-minded individuals whose reliability is unmatched.",
        "traits": ["Responsible", "Thorough", "Dependable", "Traditional"],
        "strengths": "Organization, reliability, attention to detail",
        "careers": "Accountant, Administrator, Inspector",
    },
    "ISFJ": {
        "name": "The Defender",
        "description": "Very dedicated and warm protectors, always ready to defend loved ones.",
        "traits": ["Supportive", "Reliable", "Patient", "Observant"],
        "strengths": "Attention to detail, reliability, care for others",
        "careers": "Nurse, Social Worker, Administrator",
    },
    "ESTJ": {
        "name": "The Executive",
        "description": "Excellent administrators, unsurpassed at managing things and people.",
        "traits": ["Organized", "Loyal", "Traditional", "Strong-willed"],
        "strengths": "Management, planning, decision-making",
        "careers": "Manager, Judge, Financial Officer",
    },
    "ESFJ": {
        "name": "The Consul",
        "description": "Extraordinarily caring, social and popular — always eager to help.",
        "traits": ["Caring", "Social", "Popular", "Loyal"],
        "strengths": "Social skills, cooperation, care",
        "careers": "Sales, Healthcare, Teacher",
    },
    "ISTP": {
        "name": "The Virtuoso",
        "description": "Bold and practical experimenters, masters of tools and techniques.",
        "traits": ["Practical", "Reserved", "Spontaneous", "Rational"],
        "strengths": "Mechanics, troubleshooting, calm under pressure",
        "careers": "Engineer, Pilot, Mechanic",
    },
    "ISFP": {
        "name": "The Adventurer",
        "description": "Flexible and charming artists who are always ready to explore.",
        "traits": ["Artistic", "Charming", "Sensitive", "Curious"],
        "strengths": "Creativity, empathy, aesthetics",
        "careers": "Artist, Chef, Designer",
    },
    "ESTP": {
        "name": "The Entrepreneur",
        "description": "Smart, energetic and perceptive — they truly enjoy living on the edge.",
        "traits": ["Energetic", "Perceptive", "Bold", "Direct"],
        "strengths": "Persuasion, observation, action",
        "careers": "Sales, Marketer, Athlete",
    },
    "ESFP": {
        "name": "The Entertainer",
        "description": "Spontaneous, energetic and enthusiastic entertainers who life is never boring around.",
        "traits": ["Spontaneous", "Energetic", "Enthusiastic", "Playful"],
        "strengths": "Performance, social skills, practical skills",
        "careers": "Performer, Event Planner, Sales",
    },
}


def classify_mbti(m):
    """
    Maps facial measurements to MBTI dimensions.
    Each dimension uses 1–2 facial ratios as a heuristic signal.

    m: dict from measure_features()
    Returns: (mbti_code, scores_dict)
    """

    scores = {}

    # --- I vs E: Eye openness + inter-eye distance ---
    # Wide-open eyes, close together → expressive/extrovert
    # Narrower eyes → more reserved/introvert
    eye_score = m["eye_openness"] * 10 + (1 / max(m["inter_eye_dist"], 1)) * 100
    scores["E_score"] = round(eye_score, 2)
    E = eye_score > 3.5

    # --- N vs S: Brow arch ratio ---
    # High arched brows → intuitive, imaginative
    # Flatter brows → grounded, sensing
    scores["N_score"] = round(m["brow_arch_ratio"], 3)
    N = m["brow_arch_ratio"] > 1.2

    # --- F vs T: Mouth ratio (expressiveness) ---
    # More open/expressive mouth → feeling
    # Thinner mouth → thinking/analytical
    scores["F_score"] = round(m["mouth_ratio"], 3)
    F = m["mouth_ratio"] > 0.28

    # --- P vs J: Face ratio (long vs wide) ---
    # Longer face → structured/judging
    # Wider face → flexible/perceiving
    scores["J_score"] = round(m["face_ratio"], 3)
    J = m["face_ratio"] > 1.1

    mbti = (
        ("E" if E else "I") +
        ("N" if N else "S") +
        ("F" if F else "T") +
        ("J" if J else "P")
    )

    return mbti, scores
