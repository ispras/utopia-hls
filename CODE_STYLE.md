[//]: <> (SPDX-License-Identifier: Apache-2.0)

# Utopia EDA Coding Style

We use [LLVM Coding Conventions](https://llvm.org/docs/CodingStandards.html)
with some project-specific modifications.

Some of them:

1. Use LF-ended source files (*.cpp, *.h, *.hpp, etc.);
2. Use ASCII symbols only (no Cyrillic symbols are allowed);
3. Basic indent is 2 spaces (no tabs are allowed);
4. Maximum line length is 80 (no trailing whitespaces!);
5. Do not use multiple blank lines in succession;
6. Use lowercase_underscore_separated style for names of source files;
7. Use UpperCamelCase style for names of classes/enums/structures/unions;
8. Use lowerCamelCase for names of functions/methods/objects/variables;
9. "{" symbol should be on the same line as the related operator;
10. "using namespace" is forbidden;
11. Source files should have header comments (set the `<yearnum>` here):

```cpp
//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright <yearnum>-<yearnum> ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//
```

12. All the header files should have Doxygen-formatted comments for classes:

```cpp
/**
 * \brief Implements a very useful thing.
 * \author <a href="mailto:ivanov@somemail.somedomain">Ivan Ivanov</a>
 */
```

13. All includes should be listed in the following order: 1) project's own
includes; 2) side library includes; 3) system includes. Includes should be
sorted in alphabetical order at every category.
