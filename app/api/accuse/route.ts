const SOLUTION = {
  suspect: "ilya",
  motive: "To stop exposure of resource diversion",
  method: "Engineered a controlled pressure failure"
} as const;

export async function POST(request: Request) {
  const accusation = (await request.json()) as Partial<typeof SOLUTION>;
  const correct =
    accusation.suspect === SOLUTION.suspect &&
    accusation.motive === SOLUTION.motive &&
    accusation.method === SOLUTION.method;

  return Response.json({
    correct,
    message: correct
      ? "Accusation sustained. Ilya Soren engineered the pressure failure to keep Lena from exposing the off-ledger resource diversion."
      : "The theory fractures under review. The evidence does not sustain that combination of suspect, motive, and method."
  });
}
