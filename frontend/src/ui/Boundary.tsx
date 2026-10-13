// A part of the page that fails to render says why, and the rest of the page goes on: without a boundary React
// takes the whole page down on an error. A new `reset` (other filters, another result) tries again.
import { Component, type ErrorInfo, type ReactNode } from "react";

type Props = { what: string; reset?: unknown; children: ReactNode };
type State = { error: Error | null; reset?: unknown };

export class Boundary extends Component<Props, State> {
  state: State = { error: null, reset: this.props.reset };

  static getDerivedStateFromError(error: Error): Partial<State> {
    return { error };
  }

  static getDerivedStateFromProps(props: Props, state: State): Partial<State> | null {
    return props.reset !== state.reset ? { error: null, reset: props.reset } : null;
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error(`${this.props.what} failed to render:`, error, info.componentStack);
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div className="note">
        <b>{this.props.what} could not be shown:</b> {this.state.error.message}{" "}
        <button className="btn" onClick={() => location.reload()}>Reload the page</button>
      </div>
    );
  }
}
