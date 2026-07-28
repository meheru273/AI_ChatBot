import { extractFrames, parseFrame } from "./chatStream";

// Captured verbatim from the deployed backend. sse-starlette separates frames
// with CRLF, which an earlier parser that split on "\n\n" silently dropped —
// producing an empty stream and a "connection closed" error in the UI.
const WIRE =
  'event: token\r\ndata: {"content": "Hello"}\r\n\r\n' +
  'event: token\r\ndata: {"content": "!"}\r\n\r\n';

describe("extractFrames", () => {
  test("splits CRLF-separated frames as the real server sends them", () => {
    const [frames, remainder] = extractFrames(WIRE);
    expect(frames).toHaveLength(2);
    expect(remainder).toBe("");
  });

  test("splits LF-separated frames too", () => {
    const [frames] = extractFrames('event: token\ndata: {"a":1}\n\n');
    expect(frames).toHaveLength(1);
  });

  test("holds back an unterminated frame for the next read", () => {
    const [frames, remainder] = extractFrames(
      'event: token\r\ndata: {"content": "a"}\r\n\r\nevent: tok'
    );
    expect(frames).toHaveLength(1);
    expect(remainder).toBe("event: tok");
  });

  test("rejoins a CRLF split across two network chunks", () => {
    // The first chunk ends on a bare CR; the LF opens the next one.
    let buffer = 'event: token\r\ndata: {"content": "a"}\r';
    let [frames, remainder] = extractFrames(buffer);
    expect(frames).toHaveLength(0);

    buffer = remainder + '\n\r\nevent: done\r\ndata: {"content": "a"}\r\n\r\n';
    [frames] = extractFrames(buffer);
    expect(frames).toHaveLength(2);
  });

  test("ignores blank padding between frames", () => {
    const [frames] = extractFrames('\r\n\r\nevent: token\r\ndata: {"a":1}\r\n\r\n');
    expect(frames).toHaveLength(1);
  });
});

describe("parseFrame", () => {
  test("reads the event name and JSON payload", () => {
    const [frames] = extractFrames(WIRE);
    expect(parseFrame(frames[0])).toEqual({
      event: "token",
      data: { content: "Hello" },
    });
  });

  test("tolerates a stray CR left on a line", () => {
    expect(parseFrame('event: done\r\ndata: {"content": "x"}')).toEqual({
      event: "done",
      data: { content: "x" },
    });
  });

  test("skips keep-alive comments", () => {
    expect(parseFrame(": ping")).toBeNull();
  });

  test("falls back to raw text when the payload is not JSON", () => {
    expect(parseFrame("event: token\ndata: not json")).toEqual({
      event: "token",
      data: { raw: "not json" },
    });
  });

  test("reassembles a payload spanning several data lines", () => {
    expect(parseFrame('event: token\ndata: {"content":\ndata:  "hi"}')).toEqual({
      event: "token",
      data: { content: "hi" },
    });
  });

  test("tokens from a full wire capture concatenate to the message", () => {
    const [frames] = extractFrames(WIRE);
    const text = frames
      .map(parseFrame)
      .filter((f) => f && f.event === "token")
      .map((f) => f.data.content)
      .join("");
    expect(text).toBe("Hello!");
  });
});
