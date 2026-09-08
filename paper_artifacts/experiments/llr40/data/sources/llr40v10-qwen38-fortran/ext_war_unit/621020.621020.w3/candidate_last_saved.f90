! TSVC ext_war_unit: a[i] = a[i+1] + b[i]  (i = 0 .. N-2)
!
! The read of a[i+1] is always of the ORIGINAL value: the loop writes a[i],
! and a[i+1] is only overwritten later (by iteration i+1).  So the updates
! are independent once the unit anti-dependence across a chunk boundary is
! broken: capture each chunk's boundary value a(hi+1) in a first parallel
! phase (pure reads), then update in a second parallel phase where every
! element is read before it is written by its own thread.
subroutine ext_war_unit_fp64(a, b, LEN_1D) bind(C, name="ext_war_unit_fp64")
    use, intrinsic :: iso_c_binding
    use omp_lib
    implicit none
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_double), intent(inout) :: a(LEN_1D)
    real(c_double), intent(in) :: b(LEN_1D)
    integer(c_int64_t) :: i, n, nch, base, rem, lo, hi
    integer :: t, nt
    real(c_double), allocatable :: bnd(:)

    n = LEN_1D
    if (n < 2) return

    if (n < 65536) then
        do i = 1, n - 1
            a(i) = a(i + 1) + b(i)
        end do
        return
    end if

    nt = omp_get_max_threads()
    nch = min(n - 1, int(nt, c_int64_t))
    base = (n - 1) / nch
    rem = mod(n - 1, nch)
    allocate (bnd(nch))

    !$omp parallel do schedule(static) private(lo, hi, t)
    do t = 1, int(nch)
        lo = 1 + (t - 1) * base + min(t - 1, int(rem))
        hi = lo + base - 1
        if (t <= rem) hi = hi + 1
        bnd(t) = a(hi + 1)
    end do
    !$omp end parallel do

    !$omp parallel do schedule(static) private(lo, hi, t, i)
    do t = 1, int(nch)
        lo = 1 + (t - 1) * base + min(t - 1, int(rem))
        hi = lo + base - 1
        if (t <= rem) hi = hi + 1
        do i = lo, hi - 1
            a(i) = a(i + 1) + b(i)
        end do
        a(hi) = bnd(t) + b(hi)
    end do
    !$omp end parallel do

    deallocate (bnd)
end subroutine ext_war_unit_fp64
