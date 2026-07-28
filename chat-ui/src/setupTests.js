// jest-dom adds custom jest matchers for asserting on DOM nodes.
// allows you to do things like:
// expect(element).toHaveTextContent(/react/i)
// learn more: https://github.com/testing-library/jest-dom
import '@testing-library/jest-dom';

// jsdom implements no layout, so scrollIntoView does not exist. The chat log
// calls it on every message change, which would throw in every render test.
window.HTMLElement.prototype.scrollIntoView = jest.fn();
