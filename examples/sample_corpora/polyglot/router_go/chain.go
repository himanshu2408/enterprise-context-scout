package router

import "net/http"

// Middleware wraps an http.Handler to enforce request-scoped context propagation.
type Middleware func(http.Handler) http.Handler

// Chain composes middlewares left-to-right around the terminal handler.
func Chain(handler http.Handler, middlewares ...Middleware) http.Handler {
	wrapped := handler
	for i := len(middlewares) - 1; i >= 0; i-- {
		wrapped = middlewares[i](wrapped)
	}
	return wrapped
}
