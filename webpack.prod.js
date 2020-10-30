const path = require("path");
const TerserPlugin = require("terser-webpack-plugin");

module.exports = {
  entry: {
    "canvas-to-tensor": "./ts/index.ts",
  },
  mode: "production",
  module: {
    rules: [
      {
        test: /\.ts$/i,
        use: "ts-loader",
        exclude: /node_modules/,
      },
      {
        test: /\.css$/i,
        use: ["raw-loader", "extract-loader", "css-loader"],
      },
    ],
  },
  resolve: {
    extensions: [".ts", ".js"],
  },
  optimization: {
    usedExports: true,
    minimize: true,
    minimizer: [
      new TerserPlugin({
        extractComments: true,
        terserOptions: {
          mangle: true,
          compress: {
            drop_console: true,
          },
        },
      }),
    ],
  },
  output: {
    filename: "[name].min.js",
    path: path.resolve(__dirname, "dist"),
  },
};
