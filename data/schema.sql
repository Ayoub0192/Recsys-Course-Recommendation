
CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS concepts (
    concept_id TEXT PRIMARY KEY,
    name TEXT
);

CREATE TABLE IF NOT EXISTS lessons (
    lesson_id TEXT PRIMARY KEY,
    concept_id TEXT REFERENCES concepts(concept_id),
    title TEXT
);

CREATE TABLE IF NOT EXISTS interactions (
    id SERIAL PRIMARY KEY,
    user_id TEXT REFERENCES users(user_id),
    lesson_id TEXT REFERENCES lessons(lesson_id),
    correct BOOLEAN,
    timestamp TIMESTAMP
);
