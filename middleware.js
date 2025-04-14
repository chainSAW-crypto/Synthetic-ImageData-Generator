import { NextResponse } from "next/server"

export function middleware(request) {
  const authToken = request.cookies.get("auth_token")?.value
  const { pathname } = request.nextUrl

  // Define public routes that don't require authentication
  const publicRoutes = ["/", "/login", "/register"]
  const isPublicRoute = publicRoutes.some(
    (route) =>
      pathname === route ||
      pathname.startsWith("/api/auth") ||
      pathname.startsWith("/_next") ||
      pathname.includes("favicon.ico"),
  )

  // If the user is not authenticated and trying to access a protected route
  if (!authToken && !isPublicRoute) {
    // Use 307 temporary redirect to preserve the request method
    return NextResponse.redirect(new URL(`/login?redirect=${encodeURIComponent(pathname)}`, request.url), 307)
  }

  // If the user is authenticated and trying to access login/register
  if (authToken && (pathname === "/login" || pathname === "/register")) {
    return NextResponse.redirect(new URL("/chat", request.url))
  }

  return NextResponse.next()
}

// Update the matcher to be more specific
export const config = {
  matcher: ["/((?!api/auth|_next/static|_next/image|favicon.ico).*)"],
}
