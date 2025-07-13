"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.statusCommand = statusCommand;
const commander_1 = require("commander");
function statusCommand() {
    const command = new commander_1.Command('status');
    command.description('status command');
    return command;
}
//# sourceMappingURL=status.js.map