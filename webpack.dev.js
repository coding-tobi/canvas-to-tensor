const HtmlWebpackPlugin = require("html-webpack-plugin");

module.exports = {
  entry: "./ts/index.ts",
  mode: "development",
  devtool: "inline-source-map",
  devServer: {
    port: 9000,
  },
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
  plugins: [
    new HtmlWebpackPlugin({
      inject: false,
      title: "coding-tobi.io - canvas-to-tensor",
      template: "index.ejs",
    }),
  ],
  output: {
    filename: "[name].[contenthash].js",
    chunkFilename: "[name].[contenthash].js",
  },
};
