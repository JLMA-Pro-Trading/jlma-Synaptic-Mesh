"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.stopCommand = stopCommand;
const commander_1 = require("commander");
function stopCommand() {
    const command = new commander_1.Command('stop');
    command.description('stop command');
    return command;
}
//# sourceMappingURL=stop.js.map