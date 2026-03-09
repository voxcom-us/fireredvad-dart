import 'package:flutter_test/flutter_test.dart';
import 'package:fireredvad_example/main.dart';

void main() {
  testWidgets('App renders loading state', (WidgetTester tester) async {
    await tester.pumpWidget(const FireRedVadApp());
    expect(find.text('FireRedVAD Demo'), findsOneWidget);
  });
}
