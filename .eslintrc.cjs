/* eslint config for TypeScript + ESM */
module.exports = {
  root: true,
  env: { es2022: true, node: true },
  parser: "@typescript-eslint/parser",
  plugins: ["@typescript-eslint", "import"],
  extends: [
    "eslint:recommended",
    "plugin:@typescript-eslint/recommended",
    "plugin:import/recommended",
    "plugin:import/typescript",
    "prettier"
  ],
  settings: {
    "import/resolver": {
      node: { extensions: [".js", ".ts"] },
      typescript: { project: "./tsconfig.json" }
    }
  },
  rules: {
    "import/order": [
      "warn",
      {
        "groups": [["builtin", "external"], "internal", ["parent", "sibling", "index"]],
        "newlines-between": "always"
      }
    ],
    "@typescript-eslint/consistent-type-imports": ["warn", { prefer: "type-imports" }]
  }
};