import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import App from "./App";
import * as api from "./api/chatStream";

jest.mock("./api/chatStream");

beforeEach(() => {
  jest.resetAllMocks();
  api.checkHealth.mockResolvedValue({ status: "healthy" });
});

test("shows suggestions before any conversation has started", async () => {
  render(<App />);
  expect(await screen.findByText(/Ask me about the menu/i)).toBeInTheDocument();
});

test("reports a healthy backend", async () => {
  render(<App />);
  expect(await screen.findByText("Connected")).toBeInTheDocument();
});

test("reports an unreachable backend", async () => {
  api.checkHealth.mockRejectedValue(new TypeError("Failed to fetch"));
  render(<App />);
  expect(await screen.findByText("Disconnected")).toBeInTheDocument();
});

test("renders streamed tokens as they arrive", async () => {
  api.streamChat.mockImplementation(async ({ onToken }) => {
    onToken("Japchae ", "Japchae ");
    onToken("is £5.50.", "Japchae is £5.50.");
    return "Japchae is £5.50.";
  });

  render(<App />);
  await screen.findByText("Connected");

  await userEvent.type(screen.getByRole("textbox"), "japchae?");
  await userEvent.click(screen.getByRole("button", { name: /send/i }));

  expect(await screen.findByText(/Japchae is £5.50./)).toBeInTheDocument();
});

test("sends the whole conversation, not just the latest message", async () => {
  api.streamChat.mockResolvedValue("Yes, it is.");

  render(<App />);
  await screen.findByText("Connected");

  await userEvent.type(screen.getByRole("textbox"), "Do you have japchae?");
  await userEvent.click(screen.getByRole("button", { name: /send/i }));

  await waitFor(() => expect(api.streamChat).toHaveBeenCalled());
  const sent = api.streamChat.mock.calls[0][0].messages;
  expect(sent).toEqual([{ role: "user", content: "Do you have japchae?" }]);
});

test("surfaces a mid-stream failure to the user", async () => {
  api.streamChat.mockRejectedValue(new Error("the assistant failed"));

  render(<App />);
  await screen.findByText("Connected");

  await userEvent.type(screen.getByRole("textbox"), "hello");
  await userEvent.click(screen.getByRole("button", { name: /send/i }));

  expect(await screen.findByText(/the assistant failed/i)).toBeInTheDocument();
});
