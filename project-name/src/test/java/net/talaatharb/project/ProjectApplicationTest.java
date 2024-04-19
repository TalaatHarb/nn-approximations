package net.talaatharb.project;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class ProjectApplicationTest {

	@Test
	void testProjectStarts() {
		ProjectApplication.main(null);
		assertTrue(true);
	}
}
