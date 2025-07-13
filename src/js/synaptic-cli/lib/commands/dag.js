"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.dagCommand = dagCommand;
const commander_1 = require("commander");
function dagCommand() {
    const command = new commander_1.Command('dag');
    command.description('dag command');
    return command;
}
//# sourceMappingURL=dag.js.map