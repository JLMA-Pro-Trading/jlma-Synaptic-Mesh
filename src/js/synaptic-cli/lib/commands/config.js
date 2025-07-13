"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.configCommand = configCommand;
const commander_1 = require("commander");
function configCommand() {
    const command = new commander_1.Command('config');
    command.description('config command');
    return command;
}
//# sourceMappingURL=config.js.map